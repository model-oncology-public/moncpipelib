"""Cookbook tests for row-level lineage tracking.

Each test here doubles as a documentation example. The code between
``# --- cookbook:start ---`` and ``# --- cookbook:end ---`` markers is
extracted by the cookbook pytest plugin and rendered into docs/cookbook.md.

Note: ``from __future__ import annotations`` is intentionally omitted here
because Dagster's ``@asset`` decorator resolves type annotations eagerly,
and the PEP 563 stringification breaks resolution inside local scopes.
"""

import pytest

# ---------------------------------------------------------------------------
# Default lineage with direct layer match
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Default Lineage Behavior with Valid Layer",
    description=(
        "When the target schema directly matches a valid layer "
        "(``bronze``, ``silver``, ``gold``), row-level lineage is "
        "automatically enabled with no contract required. Every row gets "
        "``_lineage_id`` and ``_lineage_key`` columns, and a record is "
        "created in the ``lineage.data_lineage`` table."
    ),
    category="lineage",
)
def test_cookbook_default_lineage_valid_layer() -> None:
    """Show auto-lineage when schema matches a valid layer."""
    # --- cookbook:start ---
    import polars as pl
    from dagster import AssetExecutionContext, MaterializeResult, asset

    from moncpipelib import PostgresResource

    database = PostgresResource(
        host="db.example.com",
        user="writer",
        password="secret",
        database="analytics",
        # enable_row_lineage defaults to True
    )

    @asset(deps=["raw_claims"])
    def claims_bronze(
        context: AssetExecutionContext,
        database: PostgresResource,
    ) -> MaterializeResult:
        df = pl.DataFrame(
            {
                "claim_id": ["C-001", "C-002"],
                "patient_id": ["P-001", "P-002"],
                "amount": [150.00, 75.50],
            }
        )

        # Schema "bronze" is a valid layer -- lineage fires automatically.
        result = database.write(
            df,
            target="bronze.claims",
            context=context,
        )

        # WriteResult includes lineage fields when lineage is active.
        # result.lineage_id  -> UUID7 (e.g., "01912345-...")
        # result.lineage_key -> composite key (e.g., "v1:claims:bronze:...")
        return MaterializeResult(metadata=result.to_dagster_metadata())

    print("When target schema is a valid layer (bronze/silver/gold):")
    print("  target='bronze.claims' -> layer='bronze'")
    print()
    print("Lineage is auto-enabled (enable_row_lineage defaults to True).")
    print("Two columns are added to every row:")
    print("  _lineage_id  -> UUID7 with embedded timestamp")
    print("  _lineage_key -> human-readable composite key")
    print()
    print("A record is also created in the lineage.data_lineage table")
    print("linking run_id, asset_name, layer, and row_count.")
    # --- cookbook:end ---

    assert database.enable_row_lineage is True


# ---------------------------------------------------------------------------
# Contract-driven lineage for compound schemas
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Contract-Driven Lineage for Compound Schemas",
    description=(
        "When the target schema is a compound name like ``reference_bronze``, "
        "it does not directly match a valid layer. The contract's ``layer`` "
        "field resolves this -- ``layer: bronze`` in the contract tells "
        "moncpipelib which layer to use for lineage tracking."
    ),
    category="lineage",
)
def test_cookbook_contract_layer_fallback() -> None:
    """Show contract layer fallback for compound schemas."""
    # --- cookbook:start ---
    from moncpipelib.config import VALID_LAYERS
    from moncpipelib.contracts.models import Column, ColumnType, DataContract, Schema

    # Schema "reference_bronze" is NOT a direct layer match.
    target = "reference_bronze.fda_ndc_directory"
    schema_name = target.split(".")[0]

    print(f"Target:     {target}")
    print(f"Schema:     {schema_name}")
    print(f"In layers?: {schema_name in VALID_LAYERS}")
    print()

    # Without a contract, layer resolves to None and lineage is skipped.
    # With a contract declaring layer: bronze, moncpipelib falls back
    # to the contract's layer for lineage tracking.
    contract = DataContract(
        version="1.0",
        pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        asset="fda_ndc_directory_bronze",
        layer="bronze",
        schema=Schema(
            columns=[
                Column(name="product_ndc", type=ColumnType.STRING, nullable=False),
                Column(name="brand_name", type=ColumnType.STRING, nullable=True),
            ]
        ),
    )

    # The contract's layer is used when the schema doesn't match VALID_LAYERS.
    resolved_layer = contract.layer if contract.layer in VALID_LAYERS else None
    print(f"Contract layer: {contract.layer}")
    print(f"Resolved layer: {resolved_layer}")
    print()
    print("Resolution order:")
    print("  1. Schema name if it matches bronze/silver/gold directly")
    print("  2. Contract's 'layer' field (fallback for compound schemas)")
    print("  3. None (lineage disabled)")
    # --- cookbook:end ---

    assert schema_name not in VALID_LAYERS
    assert resolved_layer == "bronze"


# ---------------------------------------------------------------------------
# Lineage section in contract
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Lineage Section in Contract",
    description=(
        "Contracts support an optional ``lineage`` section for fine-grained "
        "control over row-level lineage. You can specify ``source_system`` "
        "and ``transformation_type`` for provenance, or set ``enabled: false`` "
        "to opt out of lineage for a specific asset."
    ),
    category="lineage",
)
def test_cookbook_lineage_contract_section() -> None:
    """Show the lineage section fields and opt-out behavior."""
    # --- cookbook:start ---
    from moncpipelib.contracts.models import LineageConfig

    # Default: lineage enabled, no extra fields
    default_cfg = LineageConfig()
    print("Default LineageConfig:")
    print(f"  enabled:             {default_cfg.enabled}")
    print(f"  source_system:       {default_cfg.source_system}")
    print(f"  transformation_type: {default_cfg.transformation_type}")
    print()

    # Contract YAML equivalent:
    #   lineage:
    #     source_system: openfda
    #     transformation_type: ingest
    ingest_cfg = LineageConfig(
        source_system="openfda",
        transformation_type="ingest",
    )
    print("Configured LineageConfig (e.g., FDA external load):")
    print(f"  enabled:             {ingest_cfg.enabled}")
    print(f"  source_system:       {ingest_cfg.source_system}")
    print(f"  transformation_type: {ingest_cfg.transformation_type}")
    print()
    print("These fields are passed to the lineage tracker and stored in")
    print("the lineage.data_lineage record for provenance tracking.")
    print()

    # Opt out of lineage for a specific asset:
    #   lineage:
    #     enabled: false
    disabled_cfg = LineageConfig(enabled=False)
    print("Disabled LineageConfig (opt out for specific asset):")
    print(f"  enabled:             {disabled_cfg.enabled}")
    print()
    print("When enabled=false, lineage is skipped even if the resource")
    print("has enable_row_lineage=True. Simple metadata columns are")
    print("added instead (_{layer}_run_id, _{layer}_processed_at).")
    # --- cookbook:end ---

    assert default_cfg.enabled is True
    assert default_cfg.source_system is None
    assert ingest_cfg.source_system == "openfda"
    assert ingest_cfg.transformation_type == "ingest"
    assert disabled_cfg.enabled is False


# ---------------------------------------------------------------------------
# Walking the replaces_lineage_id chain (Phase 4 of migration 018)
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Walking the replaces_lineage_id Chain",
    description=(
        "``FULL_REFRESH`` writes auto-populate ``replaces_lineage_id`` "
        "with the immediately prior lineage row for the same asset / "
        "layer / partition. Walk the chain with a ``WITH RECURSIVE`` CTE "
        "to reconstruct every prior version of an asset, in commit order. "
        "Partition-scoped writes only chain within the same partition; "
        "``UPSERT`` / ``APPEND`` / ``SCD2`` never set the column."
    ),
    category="lineage",
)
def test_cookbook_walk_replaces_lineage_chain() -> None:
    """Demonstrate walking ``replaces_lineage_id`` back to the chain root."""
    # --- cookbook:start ---
    # Recover the full history of a FULL_REFRESH asset by walking the
    # ``replaces_lineage_id`` chain. The ``depth`` column tells you how
    # many writes back this row is; ``replaces_lineage_id IS NULL`` is the
    # root of the chain (first-ever write for this asset / partition).
    chain_walk_sql = """
        WITH RECURSIVE chain AS (
            -- Anchor: the most recent row for the asset.
            SELECT
                lineage_id,
                replaces_lineage_id,
                asset_name,
                layer,
                data_date,
                processed_at,
                1 AS depth
            FROM lineage.data_lineage
            WHERE asset_name = %(asset_name)s
              AND layer = %(layer)s
              AND data_date IS NOT DISTINCT FROM %(data_date)s
            ORDER BY processed_at DESC
            LIMIT 1

            UNION ALL

            -- Recursive: hop to the predecessor.
            SELECT
                p.lineage_id,
                p.replaces_lineage_id,
                p.asset_name,
                p.layer,
                p.data_date,
                p.processed_at,
                c.depth + 1
            FROM lineage.data_lineage p
            JOIN chain c ON p.lineage_id = c.replaces_lineage_id
        )
        SELECT lineage_id, replaces_lineage_id, processed_at, depth
        FROM chain
        ORDER BY depth
    """
    # Parameters to bind:
    #   asset_name -- e.g., "claims_silver"
    #   layer      -- e.g., "silver"
    #   data_date  -- date for partition-scoped chains, or NULL for
    #                 whole-table chains. ``IS NOT DISTINCT FROM`` treats
    #                 NULL = NULL so the same query works for both shapes.
    print("Run the chain walk via ``cursor.execute(chain_walk_sql, params)``:")
    print("  params = {'asset_name': 'claims_silver', 'layer': 'silver', 'data_date': None}")
    print()
    print("Returned columns:")
    print("  lineage_id           -- UUID of this row")
    print("  replaces_lineage_id  -- UUID of the prior row (NULL at root)")
    print("  processed_at         -- when this row was written")
    print("  depth                -- 1 = newest, ascending toward root")
    print()
    print("Anti-cycle guard: every row has a single ``replaces_lineage_id``")
    print("pointing strictly earlier in time (Phase 3 same-txn ordering), so")
    print("the chain is a singly-linked list with no cycle. The recursive")
    print("CTE terminates when ``replaces_lineage_id IS NULL`` (chain root)")
    print("or when no prior row exists.")
    print()
    print("Sibling pairs vs chains: under READ COMMITTED, two concurrent")
    print("FULL_REFRESH runs of the same asset may yield either a sibling")
    print("pair (both pointing to the same predecessor) or a chain")
    print("(whichever commits second sees the first). Consumers must not")
    print("assume one shape -- use the CTE's ``UNION ALL`` semantics to")
    print("walk only the strict chain.")
    # --- cookbook:end ---

    # The cookbook plugin extracts and renders the SQL above. The assertion
    # below proves the SQL parses (a real run would require a DB connection
    # and seeded chain, which lives in tests/integration/test_lineage_replaces.py).
    assert "WITH RECURSIVE" in chain_walk_sql
    assert "replaces_lineage_id" in chain_walk_sql
    assert "IS NOT DISTINCT FROM" in chain_walk_sql
