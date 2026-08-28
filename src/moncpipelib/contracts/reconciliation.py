"""Contract reconciliation logic for write configuration.

Reconciles contract sink declarations against IO Manager write configuration,
enforcing the four-way resolution pattern:
- Contract only (no explicit metadata) -> silently override IO manager default
- Metadata only -> use unchanged
- Both set, same value -> log warning (redundant config), proceed
- Both set, different values -> ContractViolationError (always fatal)
"""

from __future__ import annotations

from typing import Any

from moncpipelib.contracts.exceptions import ContractViolationError
from moncpipelib.contracts.models import DataContract
from moncpipelib.io_managers.enums import WriteMode
from moncpipelib.resources.types import LoggingContext


class ContractReconciler:
    """Reconciles contract sink fields against IO Manager write configuration.

    All methods are stateless. They accept a contract, write_config dict,
    and context (for logging), and return a reconciled value or raise
    ContractViolationError.

    Typical usage from the IO Manager::

        ContractReconciler.reconcile_write_config(contract, table_name, write_config, context)
    """

    @staticmethod
    def find_matching_sink(
        contract: DataContract | None,
        bare_table: str,
        context: LoggingContext | Any,
        *,
        target_schema: str | None = None,
    ) -> dict[str, Any] | None:
        """Find the contract sink entry matching the target table.

        Matching strategy:

        1. **Schema filter** -- when ``target_schema`` is provided, sinks that
           declare a *different* ``schema`` are excluded from all later stages
           (#405): a write targeting ``reference_gold.dim_provider`` must never
           match a sink declaring ``synthetic_gold``. Sinks without a
           ``schema`` field are never excluded, and passing
           ``target_schema=None`` disables the filter entirely.
        2. **Strict match** -- sink ``table`` field equals ``bare_table``.
        3. **Single-sink fallback** -- when strict matching finds nothing and
           exactly one schema-compatible table sink remains, return it,
           provided the contract's table sinks don't declare differing table
           names it could be confused with. This handles asset names that
           carry a layer suffix (e.g. ``fda_ndc_directory_silver``) or a
           test-isolation ``table_prefix`` that doesn't match the sink's
           ``table`` field (``fda_ndc_directory``).

        A table-name match rejected only by the schema filter logs a warning
        and returns None -- that shape means the write target and the contract
        sink disagree about the schema, and applying the sink's configuration
        (mode, keys, PII-bearing expectations) to a different schema's table
        is exactly the #405 failure mode.

        Returns the matching sink dict, or None if no match is found.

        Raises:
            ContractViolationError: If multiple sinks match the same table
                name after schema filtering.
        """
        if contract is None or not contract.sinks:
            return None

        # Collect all table-type sinks
        table_sinks: list[dict[str, Any]] = [s for s in contract.sinks if s.get("type") == "table"]

        # Stage 1: schema filter (no-op when target_schema is None)
        compatible: list[dict[str, Any]] = [
            s
            for s in table_sinks
            if target_schema is None or s.get("schema") is None or s.get("schema") == target_schema
        ]

        # Stage 2: strict match by table name
        matches: list[dict[str, Any]] = [s for s in compatible if s.get("table") == bare_table]

        if len(matches) > 1:
            schemas = [s.get("schema", "<unset>") for s in matches]
            raise ContractViolationError(
                f"Multiple contract sinks match table '{bare_table}' "
                f"(schemas: {schemas}). Each (schema, table) pair must appear "
                f"in at most one sink entry."
            )

        if matches:
            return matches[0]

        # A name match rejected only by the schema filter means the write
        # target and the contract sink disagree -- surface it, never fall
        # back to another sink.
        rejected_schemas = [
            s.get("schema")
            for s in table_sinks
            if s.get("table") == bare_table
            and target_schema is not None
            and s.get("schema") is not None
            and s.get("schema") != target_schema
        ]
        if rejected_schemas:
            if hasattr(context, "log"):
                context.log.warning(
                    f"Contract sink(s) for table '{bare_table}' declare schema(s) "
                    f"{rejected_schemas} but the write targets schema "
                    f"'{target_schema}'; the sink configuration was NOT applied. "
                    f"Check that the write target and the contract sink agree."
                )
            return None

        # Stage 3: lenient fallback -- exactly one schema-compatible sink,
        # and no differently-named table sink it could be confused with.
        if len(compatible) == 1:
            only = compatible[0]
            if len(table_sinks) == 1 or all(
                s.get("table") == only.get("table") for s in table_sinks
            ):
                return only

        return None

    @classmethod
    def reconcile_write_config(
        cls,
        contract: DataContract | None,
        bare_table: str,
        write_config: dict[str, Any],
        context: LoggingContext,
        *,
        target_schema: str | None = None,
    ) -> None:
        """Reconcile all contract sink fields against write_config, mutating in place.

        ``target_schema`` is the schema the write actually targets (when the
        caller knows it); it is threaded to ``find_matching_sink`` so a sink
        declaring a different schema is never applied (#405).
        """
        write_config["write_mode"] = cls.reconcile_sink_mode(
            contract, bare_table, write_config, context, target_schema=target_schema
        )
        write_config["primary_key"] = cls.reconcile_primary_key(
            contract, write_config, context, bare_table=bare_table, target_schema=target_schema
        )
        write_config["business_key"] = cls.reconcile_business_key(
            contract, bare_table, write_config, context, target_schema=target_schema
        )
        write_config["tracked_columns"] = cls.reconcile_tracked_columns(
            contract, bare_table, write_config, context, target_schema=target_schema
        )
        write_config["detect_deletes"] = cls.reconcile_detect_deletes(
            contract, bare_table, write_config, context, target_schema=target_schema
        )
        write_config["skip_unchanged"] = cls.reconcile_skip_unchanged(
            contract, bare_table, write_config, context, target_schema=target_schema
        )
        write_config["full_refresh_method"] = cls.reconcile_full_refresh_method(
            contract, bare_table, write_config, context, target_schema=target_schema
        )
        write_config["partition_column"] = cls.reconcile_partition_column(
            contract, bare_table, write_config, context, target_schema=target_schema
        )
        write_config["sequence_col"] = cls.reconcile_sequence_column(
            contract, bare_table, write_config, context, target_schema=target_schema
        )

    @classmethod
    def reconcile_sink_mode(
        cls,
        contract: DataContract | None,
        bare_table: str,
        write_config: dict[str, Any],
        context: LoggingContext,
        *,
        target_schema: str | None = None,
    ) -> WriteMode:
        """Reconcile the contract sink's declared mode against the resolved write mode.

        The contract sink ``mode`` field is the authoritative spec for how data should
        be written to a table. Resolution rules:

        - Contract only (no explicit asset metadata) -> silently override the IO manager
          class-level default. The contract is the single source of truth.
        - Asset metadata only -> use it unchanged (existing behaviour).
        - Both set, same value -> log a warning (redundant config) and proceed.
        - Both set, different values -> raise ``ContractViolationError`` before any write.

        Note on ``enforce_contracts=WARN``: schema-level violations are downgraded to log
        warnings in WARN mode, but ``_enforce_contract`` still returns the contract object.
        Mode conflicts are always a hard failure regardless of enforcement mode -- an
        ambiguous write mode means we cannot safely determine what to write, so the
        pipeline must fail rather than silently pick one side.
        """
        write_mode: WriteMode = write_config["write_mode"]

        sink = cls.find_matching_sink(contract, bare_table, context, target_schema=target_schema)
        if sink is None:
            return write_mode

        contract_mode_str: str | None = sink.get("mode")
        if contract_mode_str is None:
            return write_mode

        try:
            contract_mode = WriteMode(contract_mode_str)
        except ValueError as exc:
            raise ContractViolationError(
                f"Contract sink for '{bare_table}' declares unknown write mode "
                f"'{contract_mode_str}'. "
                f"Valid values: {[m.value for m in WriteMode]}"
            ) from exc

        mode_explicit: bool = write_config["write_mode_explicit"]

        if not mode_explicit:
            return contract_mode
        elif write_mode == contract_mode:
            context.log.warning(
                f"Write mode '{write_mode.value}' is declared in both the asset "
                f"metadata and the contract's sink definition for '{bare_table}'. "
                f"Remove it from one location to avoid ambiguity."
            )
            return write_mode
        else:
            raise ContractViolationError(
                f"Write mode conflict for '{bare_table}': asset metadata specifies "
                f"'{write_mode.value}' but the contract's sink declares "
                f"'{contract_mode_str}'. Resolve by removing mode from one location."
            )

    @staticmethod
    def reconcile_primary_key(
        contract: DataContract | None,
        write_config: dict[str, Any],
        context: LoggingContext,
        *,
        bare_table: str | None = None,
        target_schema: str | None = None,
    ) -> list[str] | None:
        """Reconcile contract primary key columns with the resolved primary key.

        The contract-declared primary key is the matching sink's
        ``primary_key`` field when present (the spec'd "alternative to
        schema-level ``primary_key: true``" -- ignored until #401), else the
        columns marked ``primary_key: true`` in the contract schema. The two
        may legitimately differ: sink-level names the upsert conflict key,
        schema-level often marks a surrogate identifier. Resolution against
        asset metadata follows the standard rules:

        - Contract only (no explicit asset metadata) -> silently use contract PK.
        - Asset metadata only -> use it unchanged (existing behaviour).
        - Both set, same columns (order-independent) -> log a warning (redundant
          config) and proceed.
        - Both set, different columns -> raise ``ContractViolationError``.

        Args:
            contract: The loaded DataContract, or None if no contract exists.
            write_config: Dict returned by ``_get_write_config()``.
            context: Logging context (used for ``context.log`` calls).
            bare_table: Target table name used to locate the matching sink.
                When omitted (pre-#401 callers), sink-level ``primary_key``
                is not consulted and only schema-level flags apply.

        Returns:
            The reconciled primary key column list, or None if unset.

        Raises:
            ContractViolationError: If the contract and asset metadata declare
                conflicting primary key columns.
        """
        primary_key: list[str] | None = write_config["primary_key"]

        if contract is None:
            return primary_key

        contract_pk: list[str] = []
        if bare_table is not None:
            sink = ContractReconciler.find_matching_sink(
                contract, bare_table, context, target_schema=target_schema
            )
            if sink is not None:
                sink_pk_raw = sink.get("primary_key")
                if isinstance(sink_pk_raw, str):
                    contract_pk = [sink_pk_raw]
                elif isinstance(sink_pk_raw, list):
                    contract_pk = [str(c) for c in sink_pk_raw]
        if not contract_pk:
            contract_pk = contract.get_primary_key_columns()
        if not contract_pk:
            return primary_key

        pk_explicit: bool = write_config["primary_key_explicit"]

        if primary_key is None or not pk_explicit:
            # Contract is authoritative; silently override IO manager default.
            return contract_pk
        elif sorted(primary_key) == sorted(contract_pk):
            # Both explicitly declare the same columns -- redundant configuration.
            context.log.warning(
                f"Primary key {primary_key} is declared in both the asset metadata "
                f"and the contract's column definitions. "
                f"Remove it from one location to avoid ambiguity."
            )
            return primary_key
        else:
            # Conflict between asset metadata and contract -- always fatal.
            raise ContractViolationError(
                f"Primary key conflict: asset metadata specifies {primary_key} "
                f"but the contract declares {contract_pk} "
                f"(via the sink's primary_key field or primary_key: true columns). "
                f"Resolve by removing primary_key from one location."
            )

    @classmethod
    def reconcile_business_key(
        cls,
        contract: DataContract | None,
        bare_table: str,
        write_config: dict[str, Any],
        context: LoggingContext,
        *,
        target_schema: str | None = None,
    ) -> list[str] | None:
        """Reconcile the contract sink's declared business_key against write_config.

        The contract sink ``business_key`` field is the authoritative specification
        of the SCD2 business key when present. Uses the standard four-way pattern.
        """
        business_key: list[str] | None = write_config["business_key"]

        sink = cls.find_matching_sink(contract, bare_table, context, target_schema=target_schema)
        if sink is None:
            return business_key

        contract_bk_raw: Any = sink.get("business_key")
        if contract_bk_raw is None:
            return business_key

        # Normalise to list[str] -- YAML allows a bare string or a list
        contract_bk: list[str] = (
            [contract_bk_raw]
            if isinstance(contract_bk_raw, str)
            else [str(c) for c in contract_bk_raw]
        )

        bk_explicit: bool = write_config["business_key_explicit"]

        if not bk_explicit:
            # Contract is authoritative; silently override the default (None).
            return contract_bk
        elif sorted(business_key or []) == sorted(contract_bk):
            # Both explicitly declare the same columns -- redundant configuration.
            context.log.warning(
                f"business_key {business_key} is declared in both the asset metadata "
                f"and the contract's sink definition for '{bare_table}'. "
                f"Remove it from one location to avoid ambiguity."
            )
            return business_key
        else:
            raise ContractViolationError(
                f"business_key conflict for '{bare_table}': asset metadata specifies "
                f"{business_key} but the contract's sink declares {contract_bk}. "
                f"Resolve by removing business_key from one location."
            )

    @classmethod
    def reconcile_tracked_columns(
        cls,
        contract: DataContract | None,
        bare_table: str,
        write_config: dict[str, Any],
        context: LoggingContext,
        *,
        target_schema: str | None = None,
    ) -> list[str] | None:
        """Reconcile the contract sink's declared tracked_columns against write_config.

        Uses the standard four-way pattern:
        - Contract only (no explicit metadata) -> silently override IO manager default.
        - Metadata only -> use unchanged.
        - Both set, same value -> log warning, proceed.
        - Both set, different values -> ``ContractViolationError``.
        """
        tracked_columns: list[str] | None = write_config["tracked_columns"]

        sink = cls.find_matching_sink(contract, bare_table, context, target_schema=target_schema)
        if sink is None:
            return tracked_columns

        contract_tc_raw: Any = sink.get("tracked_columns")
        if contract_tc_raw is None:
            return tracked_columns

        contract_tc: list[str] = (
            [contract_tc_raw]
            if isinstance(contract_tc_raw, str)
            else [str(c) for c in contract_tc_raw]
        )

        tc_explicit: bool = write_config["tracked_columns_explicit"]

        if not tc_explicit:
            # Contract is authoritative; silently override the default (None).
            return contract_tc
        elif sorted(tracked_columns or []) == sorted(contract_tc):
            # Both explicitly declare the same columns -- redundant configuration.
            context.log.warning(
                f"tracked_columns {tracked_columns} is declared in both the asset "
                f"metadata and the contract's sink definition for '{bare_table}'. "
                f"Remove it from one location to avoid ambiguity."
            )
            return tracked_columns
        else:
            raise ContractViolationError(
                f"tracked_columns conflict for '{bare_table}': asset metadata "
                f"specifies {tracked_columns} but the contract's sink declares "
                f"{contract_tc}. Resolve by removing tracked_columns from one "
                f"location."
            )

    @classmethod
    def reconcile_detect_deletes(
        cls,
        contract: DataContract | None,
        bare_table: str,
        write_config: dict[str, Any],
        context: LoggingContext,
        *,
        target_schema: str | None = None,
    ) -> bool:
        """Reconcile the contract sink's declared detect_deletes flag.

        Uses the standard four-way pattern. ``detect_deletes`` controls whether
        absent records are expired in SCD2 mode.
        """
        detect_deletes: bool = write_config["detect_deletes"]

        sink = cls.find_matching_sink(contract, bare_table, context, target_schema=target_schema)
        if sink is None:
            return detect_deletes

        contract_dd_raw: Any = sink.get("detect_deletes")
        if contract_dd_raw is None:
            return detect_deletes

        contract_dd: bool = bool(contract_dd_raw)
        dd_explicit: bool = write_config["detect_deletes_explicit"]

        if not dd_explicit:
            # Contract is authoritative; silently override the default (False).
            return contract_dd
        elif detect_deletes == contract_dd:
            context.log.warning(
                f"detect_deletes={detect_deletes} is declared in both the asset "
                f"metadata and the contract's sink definition for '{bare_table}'. "
                f"Remove it from one location to avoid ambiguity."
            )
            return detect_deletes
        else:
            raise ContractViolationError(
                f"detect_deletes conflict for '{bare_table}': asset metadata specifies "
                f"{detect_deletes} but the contract's sink declares {contract_dd}. "
                f"Resolve by removing detect_deletes from one location."
            )

    @classmethod
    def reconcile_skip_unchanged(
        cls,
        contract: DataContract | None,
        bare_table: str,
        write_config: dict[str, Any],
        context: LoggingContext,
        *,
        target_schema: str | None = None,
    ) -> bool:
        """Reconcile the contract sink's declared skip_unchanged flag.

        Uses the standard four-way pattern. ``skip_unchanged`` controls whether
        the upsert merge suppresses ``DO UPDATE`` for conflicting rows whose
        update columns are all unchanged (mirror issue
        model-oncology-public/moncpipelib#3). ``.get()`` reads: config dicts
        built before this key existed may not carry it.
        """
        skip_unchanged: bool = write_config.get("skip_unchanged", False)

        sink = cls.find_matching_sink(contract, bare_table, context, target_schema=target_schema)
        if sink is None:
            return skip_unchanged

        contract_su_raw: Any = sink.get("skip_unchanged")
        if contract_su_raw is None:
            return skip_unchanged

        contract_su: bool = bool(contract_su_raw)
        su_explicit: bool = write_config.get("skip_unchanged_explicit", False)

        if not su_explicit:
            # Contract is authoritative; silently override the default (False).
            return contract_su
        elif skip_unchanged == contract_su:
            context.log.warning(
                f"skip_unchanged={skip_unchanged} is declared in both the asset "
                f"metadata and the contract's sink definition for '{bare_table}'. "
                f"Remove it from one location to avoid ambiguity."
            )
            return skip_unchanged
        else:
            raise ContractViolationError(
                f"skip_unchanged conflict for '{bare_table}': asset metadata specifies "
                f"{skip_unchanged} but the contract's sink declares {contract_su}. "
                f"Resolve by removing skip_unchanged from one location."
            )

    @classmethod
    def reconcile_full_refresh_method(
        cls,
        contract: DataContract | None,
        bare_table: str,
        write_config: dict[str, Any],
        context: LoggingContext,
        *,
        target_schema: str | None = None,
    ) -> str | None:
        """Reconcile the contract sink's declared full_refresh_method override.

        Uses the standard four-way pattern, returning ``str | None``: ``None``
        means neither the ``write()`` kwarg nor the contract set it, so the
        resource-level default applies. ``full_refresh_method`` pins the
        full_refresh clear method (auto/delete/truncate) -- notably ``"delete"``
        for a target TRUNCATE cannot clear, such as one referenced by a foreign
        key (mirror issue model-oncology-public/moncpipelib#4). ``.get()``
        reads: config dicts built before this key existed may not carry it.
        """
        method: str | None = write_config.get("full_refresh_method")

        sink = cls.find_matching_sink(contract, bare_table, context, target_schema=target_schema)
        if sink is None:
            return method

        contract_frm_raw: Any = sink.get("full_refresh_method")
        if contract_frm_raw is None:
            return method

        contract_frm: str = str(contract_frm_raw)
        frm_explicit: bool = write_config.get("full_refresh_method_explicit", False)

        if not frm_explicit:
            # Contract is authoritative; silently override the default (None).
            return contract_frm
        elif method == contract_frm:
            context.log.warning(
                f"full_refresh_method={method!r} is declared in both the write() "
                f"call and the contract's sink definition for '{bare_table}'. "
                f"Remove it from one location to avoid ambiguity."
            )
            return method
        else:
            raise ContractViolationError(
                f"full_refresh_method conflict for '{bare_table}': write() specifies "
                f"{method!r} but the contract's sink declares {contract_frm!r}. "
                f"Resolve by removing full_refresh_method from one location."
            )

    @classmethod
    def reconcile_partition_column(
        cls,
        contract: DataContract | None,
        bare_table: str,
        write_config: dict[str, Any],
        context: LoggingContext,
        *,
        target_schema: str | None = None,
    ) -> str | None:
        """Reconcile the contract sink's declared partition_column.

        Uses the standard four-way pattern:
        - Contract only -> silently override IO manager default.
        - Metadata only -> use unchanged.
        - Both set, same value -> log warning, proceed.
        - Both set, different values -> ``ContractViolationError``.
        """
        partition_column: str | None = write_config["partition_column"]

        sink = cls.find_matching_sink(contract, bare_table, context, target_schema=target_schema)
        if sink is None:
            return partition_column

        contract_pc: str | None = sink.get("partition_column")
        if contract_pc is None:
            return partition_column

        pc_explicit: bool = write_config.get("partition_column_explicit", False)

        if not pc_explicit:
            # Contract is authoritative; silently override the default (None).
            return contract_pc
        elif partition_column == contract_pc:
            context.log.warning(
                f"partition_column '{partition_column}' is declared in both the asset "
                f"metadata and the contract's sink definition for '{bare_table}'. "
                f"Remove it from one location to avoid ambiguity."
            )
            return partition_column
        else:
            raise ContractViolationError(
                f"partition_column conflict for '{bare_table}': asset metadata specifies "
                f"'{partition_column}' but the contract's sink declares '{contract_pc}'. "
                f"Resolve by removing partition_column from one location."
            )

    @classmethod
    def reconcile_sequence_column(
        cls,
        contract: DataContract | None,
        bare_table: str,
        write_config: dict[str, Any],
        context: LoggingContext,
        *,
        target_schema: str | None = None,
    ) -> str | None:
        """Reconcile the contract sink's declared sequence_column.

        Uses the standard four-way pattern:
        - Contract only -> silently override the default.
        - Metadata only -> use unchanged.
        - Both set, same value -> log warning, proceed.
        - Both set, different values -> ``ContractViolationError``.

        The contract may set ``sequence_column: null`` to explicitly opt out
        of per-business-key version sequencing.
        """
        sequence_col: str | None = write_config.get("sequence_col")

        sink = cls.find_matching_sink(contract, bare_table, context, target_schema=target_schema)
        if sink is None:
            return sequence_col

        # "sequence_column" is the contract YAML key; internal key is "sequence_col"
        if "sequence_column" not in sink:
            return sequence_col

        contract_sc: str | None = sink.get("sequence_column")
        sc_explicit: bool = write_config.get("sequence_col_explicit", False)

        if not sc_explicit:
            # Contract is authoritative; silently override the default.
            return contract_sc
        elif sequence_col == contract_sc:
            if sequence_col is not None:
                context.log.warning(
                    f"sequence_column '{sequence_col}' is declared in both the asset "
                    f"metadata and the contract's sink definition for '{bare_table}'. "
                    f"Remove it from one location to avoid ambiguity."
                )
            return sequence_col
        else:
            raise ContractViolationError(
                f"sequence_column conflict for '{bare_table}': asset metadata specifies "
                f"'{sequence_col}' but the contract's sink declares '{contract_sc}'. "
                f"Resolve by removing sequence_column from one location."
            )
