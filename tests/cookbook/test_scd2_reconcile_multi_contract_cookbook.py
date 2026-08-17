"""Cookbook tests for multi-contract SCD2 reconciliation patterns.

The single-contract case is covered by the
``make_reconciliation_job`` / ``make_reconciliation_asset`` factories
in :mod:`moncpipelib.jobs`.  This example documents the multi-contract
pattern -- where a single Dagster op reconciles several SCD2 sinks
under a shared source_id -- and shows the idiomatic ``context=context``
shape on ``database.reconcile_scd2`` that mirrors
``database.write(context=context)``.

Note: ``from __future__ import annotations`` is intentionally omitted
here because Dagster decorators resolve type annotations eagerly.
"""

import pytest


@pytest.mark.cookbook(
    title="Multi-contract SCD2 reconciliation with context=context",
    description=(
        "When a single pipeline manages multiple SCD2 sinks under one "
        "``source_id`` (e.g. an NPI pipeline with separate sinks for "
        "the main registry, taxonomies, and deactivations), the "
        "``make_reconciliation_job`` factory's single-contract shape "
        "isn't a fit.  Custom ops can loop over contracts directly -- "
        "the cleanest shape is ``database.reconcile_scd2(context=context, "
        "contract=c, ...)``, which mirrors ``database.write(context=...)`` "
        "and pulls ``run_id`` (and ``asset_name`` when no contract is "
        "supplied) from the context automatically.  No string plumbing, "
        "no way to forget ``run_id``."
    ),
    category="scd2",
)
def test_cookbook_reconcile_scd2_multi_contract_with_context() -> None:
    """Document the multi-contract reconcile + period-stamp pattern."""
    # --- cookbook:start ---
    from datetime import UTC, datetime

    from dagster import Definitions, OpExecutionContext, job, op

    from moncpipelib import PostgresResource
    from moncpipelib.config import MetadataKeys

    # In practice these come from your contract loader -- one per SCD2
    # sink that shares a source_id.
    NPI_SOURCE_ID = "11111111-2222-3333-4444-555555555555"
    NPI_CONTRACTS = [
        # ("npidata", load_contract("silver/reference/nppes/npidata")),
        # ("taxonomies", load_contract("silver/reference/nppes/taxonomies")),
        # ("deactivations", load_contract("silver/reference/nppes/deactivations")),
    ]

    @op(required_resource_keys={"database"})
    def reconcile_all_npi_scd2(context: OpExecutionContext) -> None:
        database: PostgresResource = context.resources.database

        per_target: dict[str, dict[str, int]] = {}
        for label, contract in NPI_CONTRACTS:
            context.log.info(f"Reconciling {label} ({contract.asset})")
            # The idiomatic shape: pass ``context=context``.  The
            # resource extracts ``run_id`` from it so the audit row in
            # ``lineage.scd2_reconciliations`` carries the Dagster run
            # UUID, not a placeholder.  Explicit ``run_id=...`` still
            # works for ad-hoc CLI / notebook callers without a
            # Dagster context.
            result = database.reconcile_scd2(
                context=context,
                contract=contract,
                collapse_duplicates=True,
            )
            per_target[label] = {
                "rows_timeline_updated": int(result["rows_timeline_updated"]),
                "rows_collapsed": int(result["rows_collapsed"]),
                "rows_renumbered": int(result.get("rows_renumbered", 0)),
            }

        # After all reconciles complete, stamp the period registry once
        # for the shared ``source_id``.  Single-contract callers should
        # prefer ``make_reconciliation_job`` / ``make_reconciliation_asset``,
        # which encapsulate this stamping step.
        periods = database.get_registry_periods(source_id=NPI_SOURCE_ID, status="materialized")
        now = datetime.now(UTC).isoformat()
        for period in periods:
            database.update_period_metadata(
                source_id=NPI_SOURCE_ID,
                partition_key=period["partition_key"],
                metadata_updates={
                    MetadataKeys.RECONCILED_AT: now,
                    MetadataKeys.RECONCILED_BY: "reconcile_all_npi_scd2",
                },
            )

    @job
    def reconcile_all_npi_scd2_job() -> None:
        reconcile_all_npi_scd2()

    Definitions(jobs=[reconcile_all_npi_scd2_job])

    print("Multi-contract reconcile pattern:")
    print("  database.reconcile_scd2(context=context, contract=c, ...)")
    print("  - context yields run_id automatically")
    print("  - one period-registry stamp at the end covers all contracts")
    print("  - prefer make_reconciliation_job for single-contract cases")
    # --- cookbook:end ---

    # The example is illustrative -- with no real contracts in
    # NPI_CONTRACTS the @op body is a no-op loop.  Asserting the job
    # was constructed at all proves the wiring compiles.
    assert reconcile_all_npi_scd2_job.name == "reconcile_all_npi_scd2_job"
