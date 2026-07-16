"""Contract enforcement helpers extracted from PostgresResource.

These are the pure-function bodies of the resource's contract methods:

- :func:`load_contract_for_write`
- :func:`enforce_contract`
- :func:`log_validation_result`
- :func:`validate_write_config`
- :func:`validate_columns`
- :func:`validate_partition_safety`

The ``PostgresResource`` methods of the same names (``_load_contract_for_write``,
``_enforce_contract``, etc.) remain on the resource as thin wrappers.  The
wrappers supply self-derived parameters (``enforce_mode``, contract search
paths, the bound ``log_validation_result`` callable) and otherwise delegate
directly to the module-level functions here.

Keeping the wrappers on the class preserves the existing
``patch.object(PostgresResource, "_validate_columns")`` style test patches
(used ~20 times in ``tests/test_postgres_resource.py``); moving the function
bodies out shrinks the resource module without changing any caller's contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import polars as pl
import psycopg

from moncpipelib.config import LineageDefaults, SCD2Config, parse_schema_table
from moncpipelib.resources.types import _Sentinel

if TYPE_CHECKING:
    from moncpipelib.contracts.models import (
        ContractValidationSummary,
        DataContract,
        Severity,
        ValidationResult,
    )
    from moncpipelib.resources.types import LoggingContext, WriteContext


class _LogValidationCallable(Protocol):
    """Signature of the severity-aware result logger threaded into ``enforce_contract``."""

    def __call__(
        self,
        check_name: str,
        result: ValidationResult,
        wctx: LoggingContext,
        severity: Severity | None = ...,
    ) -> None: ...


def load_contract_for_write(
    *,
    contract_param: DataContract | None | _Sentinel,
    asset_name: str,
    layer: str | None,
    enforce_mode: str,
    contract_search_paths: list[Path | str] | None,
) -> DataContract | None:
    """Resolve the contract to use for a write call.

    - ``SENTINEL`` (default): auto-discover from search paths.
    - ``None``: skip contract loading entirely.
    - ``DataContract``: use as-is.
    """
    if isinstance(contract_param, _Sentinel):
        # Auto-discover
        if enforce_mode == "silent":
            return None
        from moncpipelib.contracts.loader import load_contract_for_asset

        return load_contract_for_asset(
            asset_name=asset_name,
            layer=layer,
            search_paths=contract_search_paths,
        )
    # Explicit None or DataContract
    return contract_param


def enforce_contract(
    df: pl.DataFrame,
    wctx: WriteContext,
    preloaded_contract: DataContract | None = None,
    *,
    layer: str | None = None,
    skip_table_expectations: bool = False,
    enforce_mode: str,
    contract_search_paths: list[Path | str] | None,
    log_validation_result: _LogValidationCallable,
) -> tuple[DataContract | None, ContractValidationSummary | None]:
    """Validate DataFrame against contract if one exists.

    Args:
        df: DataFrame to validate.
        wctx: Write context for logging.
        preloaded_contract: Optional pre-loaded contract to avoid re-loading.
        layer: Resolved layer name.
        skip_table_expectations: If True, skip table-level expectations
            (row_count, freshness, etc.). Used by batched writes which
            defer these to post-write SQL validation.
        enforce_mode: Resource ``enforce_contracts`` setting
            (``"silent"`` / ``"warn"`` / ``"error"``).  Threaded explicitly so
            this module does not need to know about ``PostgresResource``.
        contract_search_paths: Resolved contract search paths supplied by the
            resource wrapper.  ``None`` means "no configured paths" (the
            no-contract warning is downgraded to debug).
        log_validation_result: The severity-aware result logger.  Threaded
            explicitly (instead of imported) so test patches on
            ``PostgresResource._log_validation_result`` still take effect: the
            resource wrapper passes its bound method here.

    Returns:
        Tuple of (DataContract, ContractValidationSummary) if validated,
        or (None, None) if skipped/not found.

    Raises:
        ContractViolationError: If validation fails and enforcement is ERROR.
    """
    if enforce_mode == "silent":
        return None, None

    from moncpipelib.contracts.exceptions import ContractViolationError
    from moncpipelib.contracts.loader import load_contract_for_asset
    from moncpipelib.contracts.models import (
        CheckResultRow,
        ContractValidationSummary,
        Severity,
    )
    from moncpipelib.contracts.validators import (
        run_column_test,
        run_table_expectation,
        validate_schema,
    )

    contract: DataContract | None
    if preloaded_contract is not None:
        contract = preloaded_contract
    else:
        contract = load_contract_for_asset(
            asset_name=wctx.asset_name,
            layer=layer,
            search_paths=contract_search_paths,
        )

    if contract is None:
        if contract_search_paths:
            wctx.log.warning(
                f"No contract found for asset '{wctx.asset_name}' "
                f"(searched configured path(s)). The contract's "
                f"'asset' field must exactly match the Dagster asset name."
            )
        else:
            wctx.log.debug(f"No contract found for asset {wctx.asset_name}")
        return None, None

    wctx.log.info(f"Validating against contract: {contract.asset} (v{contract.version})")

    violations: list[ValidationResult] = []
    total_checks = 0
    passed_checks = 0
    warned_checks = 0
    warning_messages: list[str] = []
    # Migration 019 (#308) Phase 5: retain per-check audit rows so
    # the write path can persist them into
    # ``lineage.contract_validation_runs`` after the data DML.
    check_results: list[CheckResultRow] = []

    def _record(
        check_name: str,
        severity: Severity,
        result: ValidationResult,
    ) -> None:
        check_results.append(
            CheckResultRow(
                check_name=check_name,
                severity=severity.value if hasattr(severity, "value") else str(severity),
                passed=result.passed,
                failed_count=result.failed_count,
                total_count=result.total_count,
                sample_failures=result.sample_failures if not result.passed else None,
            )
        )

    # 1. Validate schema
    schema_result = validate_schema(df, contract)
    total_checks += 1
    # Schema is implicitly error-severity; failures always block under
    # ``enforce_contracts == "error"``.
    _record("schema", Severity.ERROR, schema_result)
    if schema_result.passed:
        passed_checks += 1
    else:
        violations.append(schema_result)
        log_validation_result("schema", schema_result, wctx)

    # 2. Run column tests
    for col in contract.schema.columns:
        if col.managed:
            continue
        for test in col.tests:
            result = run_column_test(
                df=df,
                column=col.name,
                test_type=test.test_type,
                parameters=test.parameters,
                when=test.when,
            )
            total_checks += 1
            _record(f"{col.name}.{test.test_type}", test.severity, result)
            if result.passed:
                passed_checks += 1
            else:
                if test.severity == Severity.ERROR:
                    violations.append(result)
                else:
                    warned_checks += 1
                    warning_messages.append(result.message)
                log_validation_result(
                    f"{col.name}.{test.test_type}",
                    result,
                    wctx,
                    severity=test.severity,
                )

    # 3. Run table expectations (skipped for batched writes - deferred to post-write SQL)
    if skip_table_expectations:
        wctx.log.debug("Table expectations deferred to post-write validation")
    for exp in [] if skip_table_expectations else contract.expectations:
        result = run_table_expectation(
            df=df,
            expectation_type=exp.expectation_type,
            parameters=exp.parameters,
        )
        total_checks += 1
        _record(exp.expectation_type, exp.severity, result)
        if result.passed:
            passed_checks += 1
        else:
            if exp.severity == Severity.ERROR:
                violations.append(result)
            else:
                warned_checks += 1
                warning_messages.append(result.message)
            log_validation_result(
                exp.expectation_type,
                result,
                wctx,
                severity=exp.severity,
            )

    # Build summary
    summary = ContractValidationSummary(
        contract_version=contract.version,
        contract_asset=contract.asset,
        status="failed" if violations else "passed",
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=len(violations),
        warned_checks=warned_checks,
        violations=[v.message for v in violations],
        warnings=warning_messages,
        check_results=check_results,
    )

    if violations and enforce_mode == "error":
        raise ContractViolationError(
            f"Contract validation failed for {wctx.asset_name}:\n"
            + "\n".join(f"  - {msg}" for msg in summary.violations),
            asset_name=wctx.asset_name,
            violations=violations,
        )

    if violations:
        wctx.log.warning(
            f"Contract validation completed with {len(violations)} error-level violation(s)"
        )
    else:
        wctx.log.info("Contract validation passed")

    return contract, summary


def log_validation_result(
    check_name: str,
    result: ValidationResult,
    wctx: LoggingContext,
    severity: Severity | None = None,
) -> None:
    """Log a validation result with appropriate severity."""
    from moncpipelib.contracts.models import Severity as _Severity

    if severity is None:
        severity = _Severity.ERROR
    status = "PASSED" if result.passed else "FAILED"
    message = f"[{severity.value.upper()}] {check_name}: {status} - {result.message}"

    if result.passed:
        wctx.log.debug(message)
    else:
        if severity == _Severity.WARN:
            wctx.log.warning(message)
        else:
            wctx.log.error(message)


def validate_write_config(
    write_config: dict[str, Any],
    df_columns: list[str],
    asset_name: str,
) -> None:
    """Validate write configuration is consistent and references valid columns.

    Raises:
        ValueError: If configuration is invalid or references missing columns.
    """
    from moncpipelib.io_managers.enums import WriteMode

    write_mode: WriteMode = write_config["write_mode"]
    primary_key: list[str] | None = write_config["primary_key"]
    update_columns: list[str] | None = write_config["update_columns"]
    partition_column: str | None = write_config["partition_column"]
    df_column_set = set(df_columns)

    if write_mode == WriteMode.UPSERT and not primary_key:
        raise ValueError(
            f"write_mode='upsert' on asset '{asset_name}' requires primary_key to be set."
        )

    # skip_unchanged is consumed only by the upsert merge; on any other mode it
    # is inert config that reads as if unchanged-row suppression were active.
    # .get(): config dicts built before this key existed may not carry it.
    if write_config.get("skip_unchanged", False) and write_mode != WriteMode.UPSERT:
        raise ValueError(
            f"skip_unchanged=True on asset '{asset_name}' is only valid with "
            f"write_mode='upsert', got '{write_mode.value}'."
        )

    # Same inert-config rationale for detect_deletes (#429): it is consumed
    # only by the SCD2 writer, and the loader's static check cannot fire when
    # the mode arrives via asset metadata / write() kwarg instead of a
    # declared sink mode. Without this backstop the flag reads as if delete
    # detection were active while nothing is expired.
    if write_config.get("detect_deletes", False) and write_mode != WriteMode.SCD2:
        raise ValueError(
            f"detect_deletes=True on asset '{asset_name}' is only valid with "
            f"write_mode='scd2', got '{write_mode.value}'."
        )

    if primary_key:
        missing_pk = [col for col in primary_key if col not in df_column_set]
        if missing_pk:
            raise ValueError(
                f"primary_key column(s) {missing_pk} not found in DataFrame for "
                f"asset '{asset_name}'. Available columns: {sorted(df_column_set)}"
            )

    if update_columns:
        missing_uc = [col for col in update_columns if col not in df_column_set]
        if missing_uc:
            raise ValueError(
                f"update_columns {missing_uc} not found in DataFrame for "
                f"asset '{asset_name}'. Available columns: {sorted(df_column_set)}"
            )

    if partition_column and partition_column not in df_column_set:
        raise ValueError(
            f"partition_column '{partition_column}' not found in DataFrame for "
            f"asset '{asset_name}'. Available columns: {sorted(df_column_set)}"
        )

    if write_mode == WriteMode.SCD2:
        business_key: list[str] | None = write_config.get("business_key")
        if not business_key:
            raise ValueError(
                f"write_mode='scd2' on asset '{asset_name}' requires business_key to be set."
            )
        missing_bk = [col for col in business_key if col not in df_column_set]
        if missing_bk:
            raise ValueError(
                f"business_key column(s) {missing_bk} not found in DataFrame for "
                f"asset '{asset_name}'. Available columns: {sorted(df_column_set)}"
            )

        tracked_cols: list[str] | None = write_config.get("tracked_columns")
        if tracked_cols:
            missing_tc = [col for col in tracked_cols if col not in df_column_set]
            if missing_tc:
                raise ValueError(
                    f"tracked_columns {missing_tc} not found in DataFrame for "
                    f"asset '{asset_name}'. Available columns: {sorted(df_column_set)}"
                )

        _scd2_v: SCD2Config = write_config.get("scd2", SCD2Config())
        scd2_temporal = {
            _scd2_v.effective_from_col,
            _scd2_v.effective_to_col,
            _scd2_v.is_current_col,
        }
        found_temporal = scd2_temporal & df_column_set
        if found_temporal:
            raise ValueError(
                f"SCD2 bookkeeping column(s) {sorted(str(c) for c in found_temporal)} should not be "
                f"in the incoming DataFrame for asset '{asset_name}'. These columns "
                f"are managed automatically."
            )


def validate_columns(
    cursor: psycopg.Cursor,
    table_name: str,
    df_columns: list[str],
    asset_name: str,
    exclude_from_table: set[str] | None = None,
) -> None:
    """Validate DataFrame columns match target table schema.

    Identity columns, generated stored columns, and lineage columns are
    excluded from validation entirely. Columns with server defaults (e.g.
    ``DEFAULT uuidv7()``) are excluded from the "missing in DataFrame"
    check but are still accepted if the DataFrame provides them.

    Raises:
        ValueError: If columns don't match, with details about mismatches.
    """
    schema, table = parse_schema_table(table_name)

    cursor.execute(
        """
        SELECT column_name, column_default, is_generated
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
          AND is_identity = 'NO'
        ORDER BY ordinal_position
        """,
        (schema, table),
    )
    rows = cursor.fetchall()

    # All writable columns: exclude generated stored columns (can't be written to)
    # Used for the "extra in DataFrame" check -- DF may include columns with defaults
    writable_columns = {row[0] for row in rows if row[2] != "ALWAYS"}

    # Required columns: writable columns that have no server default
    # Used for the "missing in DataFrame" check -- columns with defaults may be omitted
    required_columns = {row[0] for row in rows if row[1] is None and row[2] != "ALWAYS"}

    managed_columns = {LineageDefaults.ID_COLUMN, LineageDefaults.KEY_COLUMN}
    writable_columns -= managed_columns
    required_columns -= managed_columns

    if exclude_from_table:
        writable_columns -= exclude_from_table
        required_columns -= exclude_from_table

    if not writable_columns and not required_columns:
        return

    df_column_set = set(df_columns) - managed_columns

    extra_in_df = df_column_set - writable_columns
    missing_in_df = required_columns - df_column_set

    if extra_in_df or missing_in_df:
        error_parts = [f"Column mismatch for asset '{asset_name}' writing to table '{table_name}':"]
        if extra_in_df:
            error_parts.append(f"  Columns in DataFrame but not in table: {sorted(extra_in_df)}")
        if missing_in_df:
            error_parts.append(f"  Columns in table but not in DataFrame: {sorted(missing_in_df)}")
        error_parts.append(
            "\nEnsure your DataFrame schema matches the target table, "
            "or run a migration to update the table schema."
        )
        raise ValueError("\n".join(error_parts))


def validate_partition_safety(
    wctx: WriteContext,
    write_config: dict[str, Any],
    asset_name: str,
) -> None:
    """Validate that partition context + write mode combinations are safe.

    Raises:
        ContractViolationError: For unsafe combinations.
    """
    if not wctx.has_partition_key:
        return

    from moncpipelib.contracts.exceptions import ContractViolationError
    from moncpipelib.io_managers.enums import WriteMode

    write_mode: WriteMode = write_config["write_mode"]
    partition_column: str | None = write_config["partition_column"]
    primary_key: list[str] | None = write_config["primary_key"]

    # Ordered most-specific first: before #401 this guard sat below the
    # general full_refresh/scd2 guard, which also fires on scd2-without-
    # partition_column and therefore always raised first -- making this
    # message unreachable.
    if (
        write_mode == WriteMode.SCD2
        and write_config.get("detect_deletes", False)
        and partition_column is None
    ):
        raise ContractViolationError(
            f"Asset '{asset_name}' uses SCD2 with detect_deletes=True but no "
            f"partition_column is configured. Delete detection would expire "
            f"records from all partitions, not just the active partition. "
            f"Set partition_column to scope delete detection.",
            asset_name=asset_name,
        )

    if partition_column is None and write_mode in (
        WriteMode.FULL_REFRESH,
        WriteMode.SCD2,
    ):
        raise ContractViolationError(
            f"Asset '{asset_name}' is partitioned but no partition_column is configured. "
            f"Set partition_column to scope destructive operations to the active partition. "
            f"Without partition_column, write_mode='{write_mode.value}' would destroy "
            f"data from other partitions.",
            asset_name=asset_name,
        )

    if (
        write_mode == WriteMode.UPSERT
        and partition_column is not None
        and primary_key is not None
        and partition_column not in primary_key
    ):
        raise ContractViolationError(
            f"Asset '{asset_name}' uses upsert with partition_column "
            f"'{partition_column}' but the primary_key {primary_key} does not "
            f"include it. The upsert would match records across partitions. "
            f"Either add '{partition_column}' to primary_key or remove partition_column.",
            asset_name=asset_name,
        )
