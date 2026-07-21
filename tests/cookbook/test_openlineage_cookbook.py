"""Cookbook tests for OpenLineage event emission.

Each test here doubles as a documentation example. The code between
``# --- cookbook:start ---`` and ``# --- cookbook:end ---`` markers is
extracted by the cookbook pytest plugin and rendered into docs/cookbook.md.
"""

from __future__ import annotations  # noqa: I001

import json

import pytest


# ---------------------------------------------------------------------------
# Cookbook examples
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Inspect OpenLineage Custom Facets",
    description=(
        "moncpipelib attaches custom OpenLineage facets to every emitted event, "
        "carrying lineage tracking metadata, partition information, and source file "
        "details. Build facets directly to inspect the payload that will be sent "
        "to your OpenLineage backend (e.g., Marquez or DataHub)."
    ),
    category="openlineage",
)
def test_inspect_custom_facets() -> None:
    """Demonstrate building and inspecting custom OpenLineage facets."""
    # --- cookbook:start ---
    from moncpipelib.lineage import (
        DataPartitionFacet,
        MoncpipelibLineageFacet,
        SourceFileFacet,
    )

    # Lineage facet -- attached to every output dataset
    lineage_facet = MoncpipelibLineageFacet(
        lineage_id="019462a1-7b3c-7def-8abc-1234567890ab",
        lineage_key="v1:claims:bronze:2024-01-15:abc123",
        layer="bronze",
        pipeline_id="550e8400-e29b-41d4-a716-446655440000",
        is_backfill=False,
        parent_lineage_ids=[],
    )
    print("MoncpipelibLineageFacet:")
    print(json.dumps(lineage_facet.to_dict(), indent=2))

    # Partition facet -- records which date slice the data covers
    partition_facet = DataPartitionFacet(
        data_date="2024-01-15",
        data_date_start="2024-01-01",
        data_date_end="2024-01-31",
    )
    print("\nDataPartitionFacet:")
    print(json.dumps(partition_facet.to_dict(), indent=2))

    # Source file facet -- tracks where the data came from
    source_facet = SourceFileFacet(
        source_file="blob://claims-landing/claims_20240115.csv",
        source_system="azure_blob",
        file_format="csv",
    )
    print("\nSourceFileFacet:")
    print(json.dumps(source_facet.to_dict(), indent=2))
    # --- cookbook:end ---

    # Verify facets serialize correctly
    lineage_dict = lineage_facet.to_dict()
    assert lineage_dict["lineage_id"] == "019462a1-7b3c-7def-8abc-1234567890ab"
    assert lineage_dict["pipeline_id"] == "550e8400-e29b-41d4-a716-446655440000"
    assert "_schemaURL" in lineage_dict

    partition_dict = partition_facet.to_dict()
    assert partition_dict["data_date"] == "2024-01-15"

    source_dict = source_facet.to_dict()
    assert source_dict["source_file"] == "blob://claims-landing/claims_20240115.csv"


@pytest.mark.cookbook(
    title="Emit OpenLineage Events (START / COMPLETE Lifecycle)",
    description=(
        "Use ``OpenLineageEmitter`` to emit START and COMPLETE events during "
        "asset materialization. The emitter attaches schema facets (auto-derived "
        "from a Polars DataFrame), output statistics, and custom moncpipelib "
        "facets to every event. In this example ``_emit_event`` is patched so "
        "events are captured locally instead of sent over HTTP."
    ),
    category="openlineage",
)
def test_emit_start_complete_lifecycle() -> None:
    """Demonstrate the full OpenLineage START -> COMPLETE event lifecycle."""
    # --- cookbook:start ---
    from unittest.mock import patch

    import polars as pl

    from moncpipelib.lineage import OpenLineageConfig, OpenLineageEmitter

    # Configure the emitter (in production, point to Marquez / DataHub)
    config = OpenLineageConfig(
        url="http://marquez:5000",
        namespace="analytics",
    )
    emitter = OpenLineageEmitter(config)

    # Capture emitted events instead of sending them over HTTP
    captured_events: list = []

    def capture(event):  # noqa: ANN001
        captured_events.append(event)

    with patch.object(emitter, "_emit_event", side_effect=capture):
        # 1. Emit START when the asset begins materializing
        #    (pass a fixed run_id for reproducible documentation output)
        run_id = emitter.emit_start(
            job_name="claims_bronze",
            run_id="01965e00-0000-7000-8000-000000000000",
            input_datasets=["sftp.raw_claims"],
        )
        print(f"START event emitted  (run_id={run_id[:8]}...)")

        # 2. Simulate a DataFrame that was written to Postgres
        df = pl.DataFrame(
            {
                "claim_id": ["CLM-001", "CLM-002", "CLM-003"],
                "patient_id": ["PAT-00000001", "PAT-00000002", "PAT-00000003"],
                "amount": [250.00, 1200.50, 89.99],
                "status": ["approved", "pending", "denied"],
            }
        )

        # 3. Emit COMPLETE with output dataset, schema, and lineage facets
        emitter.emit_complete(
            job_name="claims_bronze",
            run_id=run_id,
            output_dataset="bronze.claims",
            row_count=len(df),
            df=df,
            lineage_id="019462a1-7b3c-7def-8abc-1234567890ab",
            lineage_key="v1:claims:bronze:2024-01-15:abc123",
            layer="bronze",
            pipeline_id="550e8400-e29b-41d4-a716-446655440000",
            source_file="blob://claims-landing/claims_20240115.csv",
            data_date="2024-01-15",
            input_datasets=["sftp.raw_claims"],
        )
        print(f"COMPLETE event emitted  (run_id={run_id[:8]}...)")

    # Inspect the captured events
    print(f"\nTotal events captured: {len(captured_events)}")

    start_event = captured_events[0]
    print("\n-- START event --")
    print(f"  Job:     {start_event.job.namespace}/{start_event.job.name}")
    print(f"  Inputs:  {[i.name for i in start_event.inputs]}")

    complete_event = captured_events[1]
    output = complete_event.outputs[0]
    print("\n-- COMPLETE event --")
    print(f"  Job:     {complete_event.job.namespace}/{complete_event.job.name}")
    print(f"  Inputs:  {[i.name for i in complete_event.inputs]}")
    print(f"  Output:  {output.namespace}/{output.name}")

    # Show schema fields auto-derived from the DataFrame
    if hasattr(output.facets.get("schema", object()), "fields"):
        schema_fields = output.facets["schema"].fields
        print(f"  Schema:  {[f.name + ' (' + f.type + ')' for f in schema_fields]}")

    # Show row count
    stats = output.facets.get("outputStatistics")
    if stats:
        print(f"  Rows:    {stats.rowCount}")

    # Show custom moncpipelib facets
    lineage_facet = output.facets.get("moncpipelibLineage")
    if lineage_facet:
        print(f"  Lineage: id={lineage_facet.lineage_id}, layer={lineage_facet.layer}")
        print(f"           pipeline_id={lineage_facet.pipeline_id}")
    # --- cookbook:end ---

    # Verify event structure
    assert len(captured_events) == 2
    assert start_event.job.name == "claims_bronze"
    assert len(start_event.inputs) == 1
    assert output.name == "bronze.claims"
    assert len(complete_event.inputs) == 1
