"""SCD2 reconciliation helpers extracted from PostgresResource.

These are the function bodies of the resource's SCD2 reconcile surface:

- :func:`extract_reconcile_context_signals` (pure)
- :func:`build_scd2_reconciliation_metadata_payload` (pure)
- :func:`resolve_scd2_sink` (pure)
- :func:`reconcile_scd2` (takes a :class:`_SCD2ResourceProtocol`)
- :func:`reconcile_scd2_with_cursor` (takes a :class:`_SCD2ResourceProtocol`)
- :func:`auto_register_period` (takes a :class:`_SCD2ResourceProtocol`)

The PostgresResource methods of the same names (``_extract_reconcile_context_signals``,
``_build_scd2_reconciliation_metadata_payload``, ``_resolve_scd2_sink``,
``reconcile_scd2``, ``_reconcile_scd2_with_cursor``, ``_auto_register_period``)
remain on the resource as thin wrappers.  Threading the resource (typed
against :class:`_SCD2ResourceProtocol` below) preserves the
``patch.object(PostgresResource, "_auto_register_period")`` test-patch
pattern and keeps ``inspect.signature(PostgresResource.reconcile_scd2)``
byte-identical for the cookbook test that pins it.

The three behavior-sensitive surfaces called out in the phase 3 plan
(advisory-lock acquisition, ROW_NUMBER+USING-ranked collapse-DELETE shape,
auto-register failure mode) are preserved verbatim.
"""

from __future__ import annotations

import json as _json
import logging
import time
from contextlib import suppress
from datetime import date
from typing import TYPE_CHECKING, Any, Protocol

import psycopg

from moncpipelib.config import MetadataKeys, SCD2Config, parse_schema_table
from moncpipelib.resources._app_name import bind_run_id
from moncpipelib.resources.types import SENTINEL, WriteContext, _Sentinel

if TYPE_CHECKING:
    from dagster import AssetExecutionContext, OpExecutionContext

    from moncpipelib.contracts.models import DataContract
    from moncpipelib.lineage import LineageTracker


class _SCD2ResourceProtocol(Protocol):
    """Subset of :class:`PostgresResource` the SCD2 helpers reach into.

    Replaces the prior ``resource: Any`` threading so the wrapper-to-helper
    hop is type-checked end-to-end.  Three-method overlap with
    :class:`_RegistryResourceProtocol` (``_check_period_registry``,
    ``_upsert_registry_row``, ``get_connection_raw``) is intentional --
    each helper module declares exactly what it needs (interface
    segregation) rather than chaining one Protocol off the other and
    pulling in unrelated methods.
    """

    # Resource fields read by ``reconcile_scd2``.  ``table_prefix`` is a
    # plain ``str`` (default ``""`` -- the empty string is the "no prefix"
    # sentinel on the resource side); ``schema_override`` and
    # ``reconcile_work_mem`` are nullable.
    schema_override: str | None
    table_prefix: str
    reconcile_work_mem: str | None

    # The three pure-helper wrappers retained as ``@staticmethod`` on the
    # resource.  Module helpers reach back through ``resource._*`` so test
    # patches on ``PostgresResource`` still dispatch correctly.
    @staticmethod
    def _resolve_scd2_sink(contract: DataContract) -> dict[str, Any]: ...

    @staticmethod
    def _extract_reconcile_context_signals(
        context: Any,
    ) -> tuple[str | None, str | None]: ...

    @staticmethod
    def _build_scd2_reconciliation_metadata_payload(
        *,
        business_key: list[str],
        collapse_duplicates: bool,
        contract: DataContract | None,
    ) -> dict[str, Any]: ...

    # ``work_mem`` resolution + apply chain (resource methods, not wrappers).
    @staticmethod
    def _resolve_work_mem(value: str | None) -> str | None: ...

    @staticmethod
    def _apply_work_mem_local(cursor: psycopg.Cursor, value: str) -> str: ...

    # Wrapper into ``_scd2_helpers.reconcile_scd2_with_cursor`` -- ``reconcile_scd2``
    # calls it via ``self`` so test patches on the resource take effect.
    def _reconcile_scd2_with_cursor(
        self,
        cursor: psycopg.Cursor,
        *,
        target: str,
        business_key: list[str],
        scd2: SCD2Config,
        collapse_duplicates: bool,
        work_mem: str | None = ...,
    ) -> tuple[int, int, int]: ...

    # Migration 019 audit-table availability probe (resource wrapper into
    # ``_registry_helpers.check_scd2_reconciliations``).
    def _check_scd2_reconciliations(
        self,
        cursor: psycopg.Cursor,
        logger: logging.Logger | None = ...,
    ) -> bool: ...

    # Issue #363: probe ``lineage.pipeline_registry`` availability so the
    # reconcile can verify the audit row's ``pipeline_id`` FK up front.
    def _check_pipeline_registry(
        self,
        cursor: psycopg.Cursor,
        wctx: WriteContext | None = ...,
    ) -> bool: ...

    # Lineage tracker accessor (returns ``None`` until configured).
    def _get_lineage_tracker(self) -> LineageTracker | None: ...

    # Registry overlap (also declared on :class:`_RegistryResourceProtocol`).
    def _check_period_registry(
        self,
        cursor: psycopg.Cursor,
        wctx: WriteContext | None = ...,
        *,
        logger: logging.Logger | None = ...,
    ) -> bool: ...

    def _upsert_registry_row(
        self,
        cursor: Any,
        *,
        source_id: str,
        source_name: str | None,
        partition_key: str,
        effective_from: date,
        effective_to: date | None,
        source_uri: str | None,
        status: str,
        registered_by: str | None,
        run_id: str | None,
        pipeline_id: str | None,
        metadata: dict[str, Any] | None,
    ) -> None: ...

    # Connection lifecycle.
    def get_connection_raw(self) -> psycopg.Connection: ...


# ---------------------------------------------------------------------------
# Pure helpers (no resource needed)
# ---------------------------------------------------------------------------


def extract_reconcile_context_signals(
    context: Any,
) -> tuple[str | None, str | None]:
    """Return ``(run_id, asset_name)`` from a Dagster context or
    :class:`WriteContext`.

    - ``WriteContext`` has both as plain ``str`` attributes already.
    - ``AssetExecutionContext`` exposes ``run_id`` and
      ``asset_key.to_user_string()``.
    - ``OpExecutionContext`` exposes ``run_id`` but ``asset_key`` is
      a ``@property`` whose getter raises ``DagsterInvalidPropertyError``
      on a non-asset op (it does *not* return ``None`` and does *not*
      raise ``AttributeError``).  ``getattr(..., default=None)`` only
      substitutes the default on ``AttributeError``, so the access
      is wrapped in ``try`` / ``except Exception`` below.  When the
      property raises, ``asset_name`` falls through to the
      contract-driven default, which is the multi-contract reconcile
      loop case.

    Defensive type-checks (``isinstance(... , str)``) protect
    against ``MagicMock``-style auto-attribute access in test
    harnesses where ``context.run_id`` would otherwise return a
    child mock and surface as a non-string value downstream.
    """
    if isinstance(context, WriteContext):
        return context.run_id, context.asset_name

    raw_run_id = getattr(context, "run_id", None)
    run_id = raw_run_id if isinstance(raw_run_id, str) else None

    asset_name: str | None = None
    # ``getattr(..., None)`` defends against attribute-absent but
    # NOT against a descriptor whose getter raises.  Dagster's
    # ``OpExecutionContext.asset_key`` is a ``@property`` that
    # raises ``DagsterInvalidPropertyError`` on non-asset ops, so
    # the read itself has to be guarded.  See #339 and the same
    # pattern on ``context.repository_def`` in
    # ``_extract_dagster_handles`` (types.py).
    try:
        raw_asset_key = getattr(context, "asset_key", None)
    except Exception:  # noqa: BLE001 -- duck-typed property; see comment above
        raw_asset_key = None
    if raw_asset_key is not None:
        to_user_string = getattr(raw_asset_key, "to_user_string", None)
        if callable(to_user_string):
            # ``to_user_string`` is invoked on a duck-typed object that
            # may be a Dagster ``AssetKey``, a partially-constructed
            # mock, or a foreign type that happens to share the
            # attribute name.  Any of those can raise on call (e.g. a
            # Dagster ``CheckError`` on an unvalidated key, a
            # ``MagicMock`` whose ``side_effect`` is configured to
            # raise).  Degrade to ``None`` so a malformed context
            # never bubbles up through ``reconcile_scd2`` -- the
            # ``isinstance(value, str)`` guard below still gates what
            # reaches the audit row.
            try:
                value = to_user_string()
            except Exception:  # noqa: BLE001 -- duck-typed call; see comment above
                value = None
            if isinstance(value, str):
                asset_name = value

    return run_id, asset_name


def build_scd2_reconciliation_metadata_payload(
    *,
    business_key: list[str],
    collapse_duplicates: bool,
    contract: DataContract | None,
) -> dict[str, Any]:
    """Build the ``scd2_reconciliations.metadata`` JSONB payload.

    Captures per-reconcile observability the typed columns don't
    carry. Typed columns already cover ``rows_*``, ``work_mem_applied``,
    ``duration_seconds``, ``target_table``, ``pipeline_id``, and
    ``asset_name`` -- this payload carries the extras:

    - ``collapse_duplicates``: the configuration flag passed to
      this reconcile (boolean; affects which DML ran).
    - ``business_key``: the resolved business-key column list,
      useful when comparing contract-derived vs. ad-hoc reconciles
      of the same target.
    - ``contract_asset``: the contract's ``asset`` identifier when
      a contract drove this reconcile.  Distinguishes contract-
      driven runs from explicit ``target=...`` callers without
      requiring a left-join against ``pipeline_registry``.
    - ``contract_version``: the contract version string, present
      alongside ``contract_asset``.

    Returns a dict; never ``None``.  ``None`` is reserved for the
    no-contract no-config case which doesn't apply here -- every
    reconcile has at least ``business_key`` and ``collapse_duplicates``.
    """
    payload: dict[str, Any] = {
        "collapse_duplicates": collapse_duplicates,
        "business_key": list(business_key),
    }
    if contract is not None:
        # Type-strict: ``contract.asset`` / ``contract.version`` are
        # ``str`` on a real ``DataContract``, but test harnesses pass
        # ``MagicMock`` contracts whose attribute access yields child
        # mocks.  Without this guard those mocks would land in the
        # JSON payload and break ``json.dumps`` at bind time.
        if isinstance(contract.asset, str):
            payload["contract_asset"] = contract.asset
        if isinstance(contract.version, str):
            payload["contract_version"] = contract.version
    return payload


def resolve_scd2_sink(contract: DataContract) -> dict[str, Any]:
    """Find the SCD2 sink from a contract.

    Raises:
        ValueError: If no SCD2 sink, multiple SCD2 sinks, or missing
            business_key.
    """
    scd2_sinks = [s for s in contract.sinks if s.get("mode") == "scd2"]
    if len(scd2_sinks) == 0:
        raise ValueError(
            f"Contract '{contract.asset}' has no SCD2 sink. "
            f"reconcile_scd2(contract=...) requires a sink with mode='scd2'."
        )
    if len(scd2_sinks) > 1:
        raise ValueError(
            f"Contract '{contract.asset}' has {len(scd2_sinks)} SCD2 sinks. "
            f"reconcile_scd2(contract=...) requires exactly one. "
            f"Pass target and business_key explicitly."
        )
    sink = scd2_sinks[0]
    if not sink.get("business_key"):
        raise ValueError(f"Contract '{contract.asset}' SCD2 sink has no business_key.")
    return sink


# ---------------------------------------------------------------------------
# Resource-coupled helpers
# ---------------------------------------------------------------------------


def _assert_pipeline_id_registered(
    cursor: psycopg.Cursor,
    pipeline_id: str,
    target: str,
) -> None:
    """Fail fast if ``pipeline_id`` is absent from ``pipeline_registry``.

    Issue #363: :meth:`LineageTracker.write_scd2_reconciliation` inserts the
    audit row -- carrying its ``pipeline_id`` FK into
    ``lineage.pipeline_registry`` -- on the *same* cursor as the reconcile
    DML, before the commit, so a dangling FK rolls back the entire reconcile
    (collapse + timeline + renumber). Checking the FK precondition here,
    *before* the expensive DML, converts a ~52-minute wasted rollback into an
    immediate, actionable failure.

    Deliberately raises rather than degrading ``pipeline_id`` to ``NULL``:
    until log-warning visibility is in place, a noisy job failure is the
    primary alert that a pipeline was never registered (registration happens
    only on the silver write path via ``pipeline_registry_upsert_committed``).
    The caller must have confirmed both ``scd2_reconciliations`` and
    ``pipeline_registry`` exist before invoking this.
    """
    from moncpipelib.config import config

    registry_schema = config.pipeline_registry.schema_name
    registry_table = config.pipeline_registry.table_name
    cursor.execute(
        f"SELECT 1 FROM {registry_schema}.{registry_table} "  # noqa: S608
        f"WHERE pipeline_id = %s",
        (pipeline_id,),
    )
    if cursor.fetchone() is None:
        raise RuntimeError(
            f"reconcile_scd2: pipeline_id {pipeline_id!r} (target {target!r}) "
            f"is not present in {registry_schema}.{registry_table}. The "
            f"scd2_reconciliations audit row would violate "
            f"fk_scd2_reconciliations_pipeline_id and roll back the entire "
            f"reconcile, so this fails before any reconcile DML runs (issue "
            f"#363). Register the pipeline -- run the silver write path for "
            f"this target so pipeline_registry_upsert_committed populates the "
            f"registry -- then retry, or pass pipeline_id=None to reconcile "
            f"without the audit FK."
        )


def reconcile_scd2(
    resource: _SCD2ResourceProtocol,
    target: str | None = None,
    business_key: list[str] | None = None,
    *,
    contract: DataContract | None = None,
    scd2: SCD2Config | None = None,
    collapse_duplicates: bool = True,
    work_mem: str | None | _Sentinel = SENTINEL,
    context: AssetExecutionContext | OpExecutionContext | WriteContext | None = None,
    run_id: str | None = None,
    asset_name: str | None = None,
    pipeline_id: str | None = None,
) -> dict[str, int | str | float | None]:
    """Body of ``PostgresResource.reconcile_scd2``.

    See the resource method for full Args/Returns docs; the docstring there
    is preserved verbatim and is the user-facing surface.
    """
    if scd2 is None:
        scd2 = SCD2Config()
    # Resolve from contract if provided
    if contract is not None:
        sink = resource._resolve_scd2_sink(contract)
        if target is None:
            schema = resource.schema_override or sink["schema"]
            table = sink["table"]
            if resource.table_prefix:
                table = f"{resource.table_prefix}{table}"
            target = f"{schema}.{table}"
        if business_key is None:
            business_key = sink["business_key"]
        # Migration 019 (#308) Phase 6: thread pipeline_id from the
        # contract into the audit row when the caller didn't pass
        # it explicitly. Asset name defaults similarly so contract-
        # driven callers don't have to plumb both.
        if pipeline_id is None:
            pipeline_id = contract.pipeline_id
        if asset_name is None:
            asset_name = contract.asset

    if target is None:
        raise ValueError("target is required (pass explicitly or via contract)")
    if business_key is None:
        raise ValueError("business_key is required (pass explicitly or via contract)")

    # ``context=`` is the idiomatic Dagster-orchestrated shape and
    # mirrors ``database.write(context=...)``.  Explicit ``run_id``
    # / ``asset_name`` always win, but a context fills in either
    # when the caller didn't pass them.
    if context is not None:
        ctx_run_id, ctx_asset_name = resource._extract_reconcile_context_signals(context)
        if run_id is None:
            run_id = ctx_run_id
        if asset_name is None:
            asset_name = ctx_asset_name

    # Issue #334 Bug 3: ``scd2_reconciliations.run_id`` is NOT NULL
    # on the SA model, and the prior ``run_id or "reconcile_scd2"``
    # literal fallback produced rows whose natural key
    # ``(run_id, asset_name, pipeline_id, applied_at)`` could not be
    # cohorted by Dagster run.  Require an explicit identifier --
    # Dagster-orchestrated callers pass ``context=context`` (preferred)
    # or ``run_id=context.run_id``; ad-hoc callers (CLI tools,
    # notebooks) must pass a meaningful tag (e.g.
    # ``"adhoc:dim_provider_2026_05_16"``).
    if run_id is None:
        raise ValueError(
            "run_id is required for reconcile_scd2(); pass context=context "
            "from a Dagster context, or an explicit identifier for ad-hoc "
            "callers. The prior 'reconcile_scd2' literal fallback was "
            "removed in #334."
        )

    # #365: bind the run_id so the (often long-running) reconcile backend
    # opened below carries it as application_name for run-to-backend
    # correlation. Ad-hoc tags bind too -- harmless, and the reaper only
    # acts on backends matching a terminal Dagster run.
    bind_run_id(run_id)

    # SENTINEL = caller did not pass -> resource field.  Explicit None
    # from the caller bypasses the resource field and disables the bump.
    # _resolve_work_mem strips, recognizes "none"/"off"/"disabled", and
    # format-validates *before* the connection is opened so malformed
    # input fails fast without consuming a backend or holding a lock.
    raw_work_mem = resource.reconcile_work_mem if isinstance(work_mem, _Sentinel) else work_mem
    effective_work_mem = resource._resolve_work_mem(raw_work_mem)

    # Migration 019 (#308) Phase 6: measure wall-clock duration around
    # the reconcile cursor block so the audit row carries an accurate
    # ``duration_seconds`` payload. ``time.perf_counter`` is the
    # monotonic clock; safe for short / long durations.
    _t0 = time.perf_counter()

    conn = resource.get_connection_raw()
    try:
        with conn.cursor() as cursor:
            _log = logging.getLogger("moncpipelib.resources")

            # Migration 019 (#308) Phase 6: the audit row is persisted on
            # this cursor before the commit so it is atomic with the
            # reconcile DML. Silent no-op until the data-platform Alembic
            # migration applies ``scd2_reconciliations``.
            audit_enabled = resource._check_scd2_reconciliations(cursor, _log)

            # #420: test-mode lineage isolation -- an ephemeral run's
            # reconcile must not write audit rows to the shared
            # lineage.scd2_reconciliations table. (Its period-registry
            # stamps are gated in _registry_helpers.update_period_metadata.)
            from moncpipelib.config import (
                SKIP_LINEAGE_WRITES_ENV,
                skip_lineage_writes,
            )

            if audit_enabled and skip_lineage_writes():
                _log.warning(
                    "%s is set: skipping scd2_reconciliations audit row for "
                    "target=%s. Test/ephemeral isolation only (#420).",
                    SKIP_LINEAGE_WRITES_ENV,
                    target,
                )
                audit_enabled = False

            # Issue #363: that same-cursor atomicity means a dangling
            # ``pipeline_id`` FK on the audit INSERT rolls back the whole
            # reconcile at commit -- after the expensive DML has already
            # run. Verify the FK will resolve *before* spending that work
            # and fail fast with an actionable error. Gated on both tables
            # existing: if either is absent there is no FK to violate (a
            # pre-migration-019 environment runs the reconcile unguarded,
            # exactly as before). These are catalog/registry reads that take
            # no lock on the target, so running them ahead of the advisory
            # lock acquired inside ``_reconcile_scd2_with_cursor`` is safe --
            # and bailing here avoids taking that lock for a doomed reconcile.
            if (
                audit_enabled
                and pipeline_id is not None
                and resource._check_pipeline_registry(cursor)
            ):
                _assert_pipeline_id_registered(cursor, pipeline_id, target)

            rows_collapsed, rows_timeline_updated, rows_renumbered = (
                resource._reconcile_scd2_with_cursor(
                    cursor,
                    target=target,
                    business_key=business_key,
                    scd2=scd2,
                    collapse_duplicates=collapse_duplicates,
                    work_mem=effective_work_mem,
                )
            )
            duration_seconds = round(time.perf_counter() - _t0, 3)

            if audit_enabled:
                tracker = resource._get_lineage_tracker()
                if tracker is not None:
                    # Issue #334 Bug 3: ``run_id`` is required by the
                    # NOT NULL column and guarded above; the prior
                    # ``or "reconcile_scd2"`` literal is gone.
                    # ``metadata`` carries the per-reconcile extras
                    # not covered by typed columns.
                    scd2_metadata = resource._build_scd2_reconciliation_metadata_payload(
                        business_key=business_key,
                        collapse_duplicates=collapse_duplicates,
                        contract=contract,
                    )
                    tracker.write_scd2_reconciliation(
                        cursor,
                        run_id=run_id,
                        asset_name=asset_name or target,
                        target_table=target,
                        pipeline_id=pipeline_id,
                        work_mem_applied=effective_work_mem,
                        rows_collapsed=rows_collapsed,
                        rows_timeline_updated=rows_timeline_updated,
                        rows_renumbered=rows_renumbered,
                        duration_seconds=duration_seconds,
                        metadata=scd2_metadata,
                    )

            conn.commit()
    finally:
        conn.close()

    return {
        "rows_timeline_updated": rows_timeline_updated,
        "rows_collapsed": rows_collapsed,
        "rows_renumbered": rows_renumbered,
        "work_mem": effective_work_mem,
        "duration_seconds": duration_seconds,
    }


def reconcile_scd2_with_cursor(
    resource: _SCD2ResourceProtocol,
    cursor: psycopg.Cursor,
    *,
    target: str,
    business_key: list[str],
    scd2: SCD2Config,
    collapse_duplicates: bool,
    work_mem: str | None = None,
) -> tuple[int, int, int]:
    """Body of ``PostgresResource._reconcile_scd2_with_cursor``.

    Caller owns the connection and transaction lifecycle (commit / rollback /
    close).  See the resource method's docstring for the full Args/Returns
    description; the load-bearing SQL forms (advisory lock, ROW_NUMBER +
    USING-ranked collapse, ``LEAD`` + ``ROW_NUMBER`` timeline / renumber)
    are preserved verbatim and are regression-guarded by the SCD2
    integration tests.
    """
    _log = logging.getLogger("moncpipelib.resources")
    effective_from_col = scd2.effective_from_col
    effective_to_col = scd2.effective_to_col
    is_current_col = scd2.is_current_col
    hash_col = scd2.hash_col
    bk_cols_quoted = ", ".join(f'"{c}"' for c in business_key)
    bk_select = ", ".join(f'"{c}"' for c in business_key)

    # Two CTEs needed because PostgreSQL forbids nested window functions.
    # Step 1 (lagged): carry forward bk + effective_from and flag group
    #   boundaries by comparing each hash with its predecessor.
    # Step 2 (hash_groups): cumulative sum to assign group IDs.
    # Step 3 (ranked): ROW_NUMBER per (bk, grp) ordered by effective_from;
    #   the rn=1 row is the keeper, rn>1 rows are deleted via USING-join.
    # Why not "DELETE WHERE id NOT IN (SELECT id FROM keepers)": Postgres
    # cannot transform NOT IN (subquery) into anti-join semantics through a
    # CTE projection chain even when the underlying column is NOT NULL,
    # because the planner can't prove non-null-ness through the projection.
    # The result is a Filter: NOT (ANY (SubPlan)) over a Materialize of
    # keepers -- O(N x M) at production scale. The USING-ranked form plans
    # cleanly as a hash/merge/nested-loop join. See #277.
    collapse_sql = (  # noqa: S608
        f"WITH lagged AS ("
        f"    SELECT id, {bk_select},"
        f'        "{effective_from_col}",'
        f"        CASE"
        f'            WHEN "{hash_col}" = LAG("{hash_col}") OVER ('
        f"                PARTITION BY {bk_cols_quoted} "
        f'                ORDER BY "{effective_from_col}"'
        f"            ) THEN 0 ELSE 1"
        f"        END AS is_new_group"
        f"    FROM {target}"
        f"), hash_groups AS ("
        f"    SELECT id, {bk_select},"
        f'        "{effective_from_col}",'
        f"        SUM(is_new_group) OVER ("
        f"            PARTITION BY {bk_cols_quoted} "
        f'            ORDER BY "{effective_from_col}"'
        f"        ) AS grp"
        f"    FROM lagged"
        f"), ranked AS ("
        f"    SELECT id, ROW_NUMBER() OVER ("
        f"        PARTITION BY {bk_cols_quoted}, grp "
        f'        ORDER BY "{effective_from_col}"'
        f"    ) AS rn"
        f"    FROM hash_groups"
        f") DELETE FROM {target} t USING ranked"
        f" WHERE t.id = ranked.id AND ranked.rn > 1"
    )

    end_of_time = scd2.end_of_time

    timeline_sql = (  # noqa: S608
        f"WITH timeline AS ("
        f"    SELECT id,"
        f'        COALESCE(LEAD("{effective_from_col}") OVER ('
        f"            PARTITION BY {bk_cols_quoted} "
        f'            ORDER BY "{effective_from_col}"'
        f"        ), '{end_of_time}'::date) AS next_effective_from,"
        f"        (ROW_NUMBER() OVER ("
        f"            PARTITION BY {bk_cols_quoted} "
        f'            ORDER BY "{effective_from_col}" DESC'
        f"        ) = 1) AS should_be_current"
        f"    FROM {target}"
        f") UPDATE {target} t"
        f' SET "{effective_to_col}" = tl.next_effective_from,'
        f'    "{is_current_col}" = tl.should_be_current'
        f" FROM timeline tl"
        f" WHERE t.id = tl.id"
        f'  AND (t."{effective_to_col}" IS DISTINCT FROM tl.next_effective_from'
        f'       OR t."{is_current_col}" IS DISTINCT FROM tl.should_be_current)'
    )

    rows_collapsed = 0
    rows_timeline_updated = 0
    rows_renumbered = 0

    # Serialize concurrent reconcile_scd2 invocations against the same
    # target. Tx-scoped lock auto-releases on commit/rollback. Per-target
    # via hashtext so reconciles against unrelated tables don't block each
    # other. Lock does NOT serialize against ongoing database.write() SCD2
    # writes -- only against other reconciles. See #278.
    cursor.execute(
        "SELECT pg_advisory_xact_lock(hashtext(%s))",
        (target,),
    )

    # Per-tx work_mem bump for the window-function sorts.  Reverts on
    # commit/rollback. See #294.  INFO-level on both branches so operators
    # reading raw run logs can answer "what work_mem did this run use?"
    # without DEBUG logging or pg_stat_activity inspection (#306).
    if work_mem is not None:
        canonical = resource._apply_work_mem_local(cursor, work_mem)
        _log.info(
            "reconcile_scd2: per-tx work_mem set to %s (canonical %s) for target %s",
            work_mem,
            canonical,
            target,
        )
    else:
        _log.info(
            "reconcile_scd2: per-tx work_mem override skipped, using cluster default for target %s",
            target,
        )

    if collapse_duplicates:
        cursor.execute(collapse_sql)
        rows_collapsed = cursor.rowcount

    cursor.execute(timeline_sql)
    rows_timeline_updated = cursor.rowcount

    # Renumber sequence column so seq_id is monotonic
    # per business key across the stitched timeline.
    if scd2.sequence_col is not None:
        schema, bare_table = parse_schema_table(target)
        cursor.execute(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
              AND column_name = %s
            """,
            (schema, bare_table, scd2.sequence_col),
        )
        if cursor.fetchone() is not None:
            renumber_sql = (  # noqa: S608
                f"UPDATE {target} t"
                f' SET "{scd2.sequence_col}" = numbered.new_seq'
                f" FROM ("
                f"    SELECT id,"
                f"        ROW_NUMBER() OVER ("
                f"            PARTITION BY {bk_cols_quoted}"
                f'            ORDER BY "{effective_from_col}"'
                f"        ) AS new_seq"
                f"    FROM {target}"
                f") numbered"
                f" WHERE t.id = numbered.id"
                f'  AND t."{scd2.sequence_col}"'
                f" IS DISTINCT FROM numbered.new_seq"
            )
            cursor.execute(renumber_sql)
            rows_renumbered = cursor.rowcount

    return (rows_collapsed, rows_timeline_updated, rows_renumbered)


def auto_register_period(
    resource: _SCD2ResourceProtocol,
    conn: psycopg.Connection,
    loaded_contract: DataContract | None,
    effective_date: date | None,
    wctx: WriteContext,
    source_id: str | None = None,
    source_uri: str | None = None,
) -> None:
    """Body of ``PostgresResource._auto_register_period``.

    Auto-register the current period after a successful write.  See the
    resource wrapper for the full docstring; the failure-mode invariant
    (catch ``Exception``, warn, do not raise) is preserved verbatim and
    is the load-bearing behavior the ``test_auto_register_period_failure_warns``
    test pins.
    """
    if loaded_contract is None:
        return

    from moncpipelib.config import config

    registry_schema = config.period_registry.schema_name
    registry_table = config.period_registry.table_name

    # --- Bronze path: data_source present + effective_date matches ---
    if loaded_contract.data_source is not None and effective_date is not None:
        from moncpipelib.contracts.models import FromIngestTemplate

        data_source = loaded_contract.data_source
        pipeline_id = loaded_contract.pipeline_id if loaded_contract else None

        # Bronze, from_ingest periods. The early invariant check in
        # database.write(...) guarantees source_uri and partition_keys
        # are populated when we reach this branch.
        if isinstance(data_source.periods, FromIngestTemplate):
            if not wctx.partition_keys:
                # Defensive guard: should be unreachable because
                # database.write(...) raises ValueError upfront. Log and
                # skip rather than crash if a non-public caller bypasses
                # validation.
                wctx.log.warning(
                    f"Skipping from_ingest period registration for source "
                    f"{data_source.source_name!r}: missing partition context"
                )
                return
            partition_key = wctx.partition_keys[0]
            try:
                with conn.cursor() as reg_cursor:
                    if not resource._check_period_registry(reg_cursor, wctx):
                        return
                    resource._upsert_registry_row(
                        reg_cursor,
                        source_id=data_source.source_id,
                        source_name=data_source.source_name,
                        partition_key=partition_key,
                        effective_from=effective_date,
                        effective_to=None,
                        source_uri=source_uri,
                        status="materialized",
                        registered_by=wctx.asset_name,
                        run_id=wctx.run_id,
                        pipeline_id=pipeline_id,
                        metadata=None,
                    )
                    conn.commit()
                wctx.log.info(
                    f"Registered from_ingest period {partition_key} for source "
                    f"{data_source.source_id} as materialized"
                )
            except Exception as reg_err:
                wctx.log.warning(f"Failed to register from_ingest period: {reg_err}")
                with suppress(Exception):
                    conn.rollback()
            return

        # Bronze, enumerated periods. Find matching period by effective_date.
        matched_period = None
        for period in data_source.periods:
            if period.effective_from == effective_date:
                matched_period = period
                break

        if matched_period is None:
            return

        partition_key = matched_period.partition_key or effective_date.isoformat()

        try:
            with conn.cursor() as reg_cursor:
                if not resource._check_period_registry(reg_cursor, wctx):
                    return
                resource._upsert_registry_row(
                    reg_cursor,
                    source_id=data_source.source_id,
                    source_name=data_source.source_name,
                    partition_key=partition_key,
                    effective_from=matched_period.effective_from,
                    effective_to=matched_period.effective_to,
                    source_uri=matched_period.source,
                    status="materialized",
                    registered_by=wctx.asset_name,
                    run_id=wctx.run_id,
                    pipeline_id=pipeline_id,
                    metadata=None,
                )
                conn.commit()
            wctx.log.info(
                f"Registered period {partition_key} for source "
                f"{data_source.source_id} as materialized"
            )
        except Exception as reg_err:
            wctx.log.warning(f"Failed to register period: {reg_err}")
            with suppress(Exception):
                conn.rollback()
        return

    # --- Silver path: stamp silver_materialized when partition context available ---
    if wctx.has_partition_key and wctx.partition_keys:
        partition_key = wctx.partition_keys[0]

        try:
            with conn.cursor() as reg_cursor:
                if not resource._check_period_registry(reg_cursor, wctx):
                    return

                # Resolve source_id: explicit parameter > partition_key lookup
                resolved_source_id = source_id
                if resolved_source_id is None:
                    resolve_sql = (  # noqa: S608
                        f"SELECT source_id FROM {registry_schema}.{registry_table} "
                        f"WHERE partition_key = %s AND status = 'materialized' "
                        f"LIMIT 1"
                    )
                    reg_cursor.execute(resolve_sql, (partition_key,))
                    row = reg_cursor.fetchone()
                    if row is not None:
                        resolved_source_id = row[0]

                if resolved_source_id is not None:
                    from datetime import UTC, datetime

                    metadata_updates = {
                        MetadataKeys.SILVER_MATERIALIZED_AT: datetime.now(UTC).isoformat(),
                        MetadataKeys.SILVER_MATERIALIZED_BY: wctx.asset_name,
                        MetadataKeys.SILVER_RUN_ID: wctx.run_id,
                    }
                    merge_sql = (  # noqa: S608
                        f"UPDATE {registry_schema}.{registry_table} "
                        f"SET metadata = COALESCE(metadata, '{{}}'::jsonb) || %s::jsonb "
                        f"WHERE source_id = %s AND partition_key = %s"
                    )
                    reg_cursor.execute(
                        merge_sql,
                        (_json.dumps(metadata_updates), resolved_source_id, partition_key),
                    )
                    conn.commit()
                    wctx.log.info(
                        f"Stamped silver_materialized for partition {partition_key} "
                        f"(source {resolved_source_id})"
                    )
        except Exception as reg_err:
            wctx.log.warning(f"Failed to stamp silver metadata: {reg_err}")
            with suppress(Exception):
                conn.rollback()
