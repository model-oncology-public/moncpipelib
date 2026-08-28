"""Cookbook: from_ingest period registry registration for bronze writes.

Walks through how a bronze asset whose contract uses
``FromIngestTemplate``-typed periods passes ``source_uri`` to
``database.write(...)`` so ``lineage.period_registry`` records the
resolved blob path. Without this, downstream silver discovery via
``build_partitions_from_registry`` cannot resolve the partition.
"""

from __future__ import annotations

import inspect

import pytest


@pytest.mark.cookbook(
    title="from_ingest period registry registration",
    description=(
        "Bronze writes against a from_ingest source must pass source_uri to "
        "database.write(...) so the period registry records the resolved blob "
        "path. Demonstrates the bronze caller pattern, the silver discovery "
        "path it unlocks, and the ValueError that fires if source_uri or the "
        "Dagster partition context is missing."
    ),
    category="period_registry",
)
def test_from_ingest_period_registry_registration() -> None:
    """Show the bronze + silver lifecycle for a from_ingest source."""
    # --- cookbook:start ---
    from moncpipelib import (
        build_partitions_from_registry,
        period_registry_sensor,
        resolve_source_for_partition,
    )

    SOURCE_UUID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    # -- Step 1: source.yaml + ingest.yaml --
    #
    # The downstream consumer source declares its periods as from_ingest --
    # one period is materialized per ingest-discovered partition rather than
    # being enumerated at design time.
    #
    #   # rxnorm_full.source.yaml
    #   source_id: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    #   name: rxnorm_full
    #   ingest_source: rxnorm_full_ingest
    #   periods:
    #     mode: from_ingest
    #     source: "rxnorm_full_{release_version}.zip"
    #     effective_from_field: effective_from
    #
    # The matching ingest.yaml lands per-partition manifests under a
    # known prefix; resolve_source_for_partition() reads the manifest at
    # bronze materialization time and returns a BlobRef pointing at the
    # actual file.

    # -- Step 2: bronze asset --
    #
    # The bronze asset:
    #   1. Resolves the per-partition blob via resolve_source_for_partition
    #   2. Reads its effective_from from the manifest
    #   3. Calls database.write(..., source_uri=blob.path) so the registry
    #      row records the actual blob path the data was loaded from.
    #
    #   from moncpipelib.ingest.types import BlobRef
    #
    #   @asset(partitions_def=BRONZE_PARTITIONS)
    #   def bronze__rxnorm_full(
    #       context,
    #       database: PostgresResource,
    #       blob: BlobStorageResource,
    #       corpus: ContractCorpus,
    #   ):
    #       source = corpus.get_source("rxnorm_full")
    #       refs = resolve_source_for_partition(
    #           source, context.partition_key, corpus, blob
    #       )
    #       (ref,) = refs
    #       assert isinstance(ref, BlobRef)
    #       df = read_zip_to_dataframe(blob.open(ref.path))
    #       effective_from = parse_manifest_effective_from(...)
    #       return database.write(
    #           df,
    #           target="reference_bronze.rxnorm_full",
    #           context=context,
    #           effective_date=effective_from,
    #           source_uri=ref.path,  # <-- required for from_ingest
    #       )
    #
    # If you forget source_uri, database.write(...) raises ValueError
    # before any SQL runs:
    #
    #   ValueError: database.write(...) requires source_uri for from_ingest
    #   source 'rxnorm_full': pass the resolved blob path obtained from
    #   resolve_source_for_partition(...)
    #
    # The same ValueError fires if the asset is materialized without a
    # Dagster partition context.

    # -- Step 3: registry row --
    #
    # After a successful bronze write, lineage.period_registry contains:
    #
    #   source_id      | aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee
    #   source_name    | rxnorm_full
    #   partition_key  | 2025-08-01      (from context.partition_key)
    #   effective_from | 2025-08-01      (from effective_date)
    #   effective_to   | NULL            (FromIngestTemplate has no end)
    #   source_uri     | <BlobRef.path>  (caller-supplied)
    #   status         | materialized
    #   metadata       | {}              (no per-partition metadata yet)

    # -- Step 4: silver discovery --
    #
    # build_partitions_from_registry queries the registry; once the bronze
    # row above lands, the silver asset's DynamicPartitionsDefinition
    # picks it up. period_registry_sensor fires per-partition RunRequests
    # to materialize silver.
    SILVER_PARTITIONS = build_partitions_from_registry(SOURCE_UUID)
    print(f"Silver partitions def: {SILVER_PARTITIONS.name}")

    #   silver_job = define_asset_job("silver_rxnorm_job", selection=[silver__rxnorm])
    #   rxnorm_sensor = period_registry_sensor(
    #       source_id=SOURCE_UUID,
    #       target_job=silver_job,
    #       partitions_def=SILVER_PARTITIONS,
    #   )

    # -- Step 5: silver write --
    #
    # Silver auto-stamps silver_materialized_at via source_id (no source_uri
    # needed; that's bronze-only).
    #
    #   @asset(partitions_def=SILVER_PARTITIONS)
    #   def silver__rxnorm(context, database: PostgresResource):
    #       df = database.read_batched(
    #           f"SELECT * FROM reference_bronze.rxnorm_full "
    #           f"WHERE load_period = '{context.partition_key}'"
    #       )
    #       return database.write(
    #           df,
    #           target="reference_silver.rxnorm",
    #           context=context,
    #           source_id=SOURCE_UUID,
    #       )

    # -- Why source_uri is required (and what the template alone cannot do) --
    #
    # FromIngestTemplate.source is a blob-relative path or glob under the
    # ingest prefix (e.g. "rxnorm_full_{release_version}.zip"). It does
    # not include the storage account, container, or scheme, so it cannot
    # stand in as a URI for audit traceability. The resolved BlobRef.path
    # returned by resolve_source_for_partition is the high-fidelity value;
    # passing it through to source_uri is what populates the registry's
    # audit trail correctly.
    # --- cookbook:end ---

    # Sanity-check that the public surface this cookbook references exists.
    assert callable(resolve_source_for_partition)
    assert callable(period_registry_sensor)
    assert SILVER_PARTITIONS.name == f"periods_{SOURCE_UUID.replace('-', '_')}"

    sig = inspect.signature(resolve_source_for_partition)
    assert list(sig.parameters.keys()) == ["source", "partition_key", "corpus", "blob"]

    # The database.write signature must expose source_uri.
    from moncpipelib.resources.postgres import PostgresResource

    write_sig = inspect.signature(PostgresResource.write)
    assert "source_uri" in write_sig.parameters
