"""Cookbook: End-to-End SCD2 Lifecycle with Registry Sensors.

Comprehensive walkthrough of the registry-driven SCD2 lifecycle across
code locations: bronze loads, silver sensors, reconciliation, gold gating.
"""

from __future__ import annotations

import inspect

import pytest


@pytest.mark.cookbook(
    title="End-to-End SCD2 Lifecycle with Registry Sensors",
    description=(
        "Complete walkthrough of the registry-driven SCD2 lifecycle across "
        "code locations. Bronze loads register periods, silver sensors discover "
        "them, reconciliation stitches the timeline, and gold sensors gate on "
        "reconciliation completion. The period registry's metadata JSONB field "
        "tracks every lifecycle transition."
    ),
    category="scd2_lifecycle",
)
def test_scd2_full_lifecycle() -> None:
    """Show the full bronze -> silver -> reconcile -> gold lifecycle."""
    # --- cookbook:start ---
    from moncpipelib import (
        build_partitions_from_registry,
        make_reconciliation_asset,
        make_reconciliation_bundle,
        period_registry_sensor,
        reconciliation_sensor,
        registry_sensor,
        scd2_registry_sensor,
    )

    # -- Lifecycle Flow --
    #
    #   Bronze (x10 partitions)
    #     |  auto-registers periods as 'materialized'
    #     v
    #   period_registry_sensor
    #     |  discovers new partition keys, adds to DynamicPartitionsDefinition
    #     v
    #   Silver (x10 partitions)
    #     |  auto-stamps silver_materialized_at via source_id param
    #     v
    #   reconciliation_sensor
    #     |  waits for ALL periods to be silver-materialized
    #     v
    #   Reconciliation Asset
    #     |  reconcile_scd2() stitches timeline, stamps reconciled_at
    #     v
    #   scd2_registry_sensor
    #     |  gates on reconciled_at for each period
    #     v
    #   Gold (x10 partitions)
    #     reads correctly reconciled silver data

    SOURCE_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

    # -- Step 1: Data Source YAML --
    #
    # Your *.source.yaml defines the periods and partition keys:
    #
    #   source_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    #   name: cms_asp_ndc_crosswalk
    #   periods:
    #     - partition_key: "2024-Q1"
    #       effective_from: "2024-01-01"
    #     - partition_key: "2024-Q2"
    #       effective_from: "2024-04-01"

    # -- Step 2: Bronze Layer --
    #
    # Use build_partitions_from_periods() for static periods, or
    # build_partitions_from_registry() for registry-backed dynamic partitions.
    #
    #   @asset(partitions_def=BRONZE_PARTITIONS)
    #   def bronze_cms_asp(context, database: PostgresResource):
    #       period = get_period_for_partition(DATA_SOURCE, context.partition_key)
    #       df = download_from_source(period.effective_from)
    #       return database.write(
    #           df, target="reference_bronze.cms_asp",
    #           context=context,
    #           effective_date=period.effective_from,  # auto-registers period
    #       )

    SILVER_PARTITIONS = build_partitions_from_registry(SOURCE_UUID)
    print(f"Silver partitions def: {SILVER_PARTITIONS.name}")

    # -- Step 3: Silver Discovery --
    #
    # period_registry_sensor discovers new bronze periods and triggers silver.
    # Internally uses registry_sensor(ready_when=always, trigger_mode="per_partition").
    #
    #   silver_job = define_asset_job("silver_cms_asp_job", selection=[silver_cms_asp])
    #   cms_sensor = period_registry_sensor(
    #       source_id=SOURCE_UUID,
    #       target_job=silver_job,
    #       partitions_def=SILVER_PARTITIONS,
    #   )

    # -- Step 4: Silver Write --
    #
    # Silver asset writes with source_id for auto-stamping:
    #
    #   @asset(partitions_def=SILVER_PARTITIONS)
    #   def silver_cms_asp(context, database: PostgresResource):
    #       period = get_period_from_registry(database, SOURCE_UUID, context.partition_key)
    #       df = database.read_batched(
    #           f"SELECT * FROM reference_bronze.cms_asp "
    #           f"WHERE load_period = '{period.partition_key}'"
    #       )
    #       return database.write(
    #           df, target="reference_silver.cms_asp",
    #           context=context,
    #           effective_date=period.effective_from,
    #           source_id=SOURCE_UUID,  # stamps silver_materialized_at
    #       )

    # -- Step 5: Reconciliation Trigger --
    #
    # reconciliation_sensor waits for ALL periods to have silver_materialized_at,
    # then fires a single RunRequest. Uses trigger_mode="all".
    #
    # Option A: manual job + sensor setup
    #   reconcile_job = make_reconciliation_job(contract, SOURCE_UUID)
    #   reconcile_sensor = reconciliation_sensor(
    #       source_id=SOURCE_UUID,
    #       target_job=reconcile_job,
    #   )
    #
    # Option B: use make_reconciliation_bundle() for all three at once
    #   asset, sensor, job = make_reconciliation_bundle(
    #       contract=silver_contract,
    #       source_id=SOURCE_UUID,
    #   )

    # -- Step 6: Gold Gating --
    #
    # scd2_registry_sensor gates on reconciled_at for each period.
    # Uses trigger_mode="per_partition" with ready_when=reconciled_at.
    #
    #   GOLD_PARTITIONS = build_partitions_from_registry(SOURCE_UUID)
    #   gold_job = define_asset_job("gold_cms_asp_job", selection=[gold_cms_asp])
    #   gold_sensor = scd2_registry_sensor(
    #       source_id=SOURCE_UUID,
    #       target_job=gold_job,
    #       partitions_def=GOLD_PARTITIONS,
    #   )

    # -- Registry Metadata at Each Stage --
    #
    # After bronze:   {"status": "materialized"}
    # After silver:   {"status": "materialized", "silver_materialized_at": "2026-03-30T..."}
    # After recon:    {"status": "materialized", "silver_materialized_at": "...",
    #                  "reconciled_at": "2026-03-30T...", "rows_timeline_updated": 42}
    #
    # Observability query:
    #   SELECT source_id, partition_key, status,
    #          metadata->>'silver_materialized_at' AS silver_at,
    #          metadata->>'reconciled_at' AS reconciled_at
    #   FROM lineage.period_registry
    #   WHERE source_id = '<uuid>'
    #   ORDER BY partition_key;

    # Verify all factories are importable and have correct signatures
    print()
    print("Factory signatures:")

    sig = inspect.signature(registry_sensor)
    print(f"  registry_sensor: {list(sig.parameters.keys())}")

    sig = inspect.signature(period_registry_sensor)
    print(f"  period_registry_sensor: {list(sig.parameters.keys())}")

    sig = inspect.signature(reconciliation_sensor)
    print(f"  reconciliation_sensor: {list(sig.parameters.keys())}")

    sig = inspect.signature(scd2_registry_sensor)
    print(f"  scd2_registry_sensor: {list(sig.parameters.keys())}")

    sig = inspect.signature(make_reconciliation_asset)
    print(f"  make_reconciliation_asset: {list(sig.parameters.keys())}")

    sig = inspect.signature(make_reconciliation_bundle)
    print(f"  make_reconciliation_bundle: {list(sig.parameters.keys())}")

    print()
    print("Registry metadata lifecycle:")
    print("  1. Bronze writes  -> status='materialized'")
    print("  2. Silver writes  -> metadata.silver_materialized_at stamped")
    print("  3. Recon sensor   -> fires when all silver-ready")
    print("  4. Recon asset    -> reconcile_scd2() + stamps reconciled_at")
    print("  5. Gold sensor    -> fires per partition when reconciled_at present")
    # --- cookbook:end ---

    # Verify all imports worked
    assert callable(registry_sensor)
    assert callable(period_registry_sensor)
    assert callable(reconciliation_sensor)
    assert callable(scd2_registry_sensor)
    assert callable(make_reconciliation_asset)
    assert callable(make_reconciliation_bundle)
    assert SILVER_PARTITIONS.name == f"periods_{SOURCE_UUID.replace('-', '_')}"
