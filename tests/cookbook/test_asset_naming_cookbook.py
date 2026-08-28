"""Cookbook tests for asset naming conventions and contract resolution.

Shows the recommended patterns for naming Dagster assets, structuring
multi-component AssetKeys, and matching contracts to assets -- including
multi-client pipelines where the same logical pipeline runs per client.

Note: ``from __future__ import annotations`` is intentionally omitted here
because Dagster's ``@asset`` decorator resolves type annotations eagerly,
and the PEP 563 stringification breaks resolution inside local scopes.
"""

from pathlib import Path

import pytest


@pytest.mark.cookbook(
    title="Asset Naming: Single-Client Pipeline",
    description=(
        "The simplest pattern: flat asset names with layer as a suffix or prefix. "
        "The contract ``asset`` field must exactly match what Dagster sees. "
        "Use ``AssetKey`` with multiple components for UI hierarchy without "
        "changing the logical asset name. "
        "**Multi-client pipelines**: when the same pipeline runs per client, "
        "use ``{client}_{pipeline}`` prefixed names instead -- see the "
        '"Asset Naming: Multi-Client Pipelines" example below.'
    ),
    category="contracts",
)
def test_single_client_naming(tmp_path: Path) -> None:
    """Show single-client asset naming with contract resolution."""
    # Create a contract with a flat asset name
    (tmp_path / "claims_silver.contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "550e8400-e29b-41d4-a716-446655440000"
asset: claims_silver
layer: silver
schema:
  columns:
    - name: claim_id
      type: string
      nullable: false
      pii: false
    - name: amount
      type: decimal
      nullable: false
      pii: false
"""
    )

    # --- cookbook:start ---
    from moncpipelib.contracts import load_contract_for_asset

    # Contract asset field is the flat Dagster asset name
    contract = load_contract_for_asset("claims_silver", search_paths=[tmp_path])
    print(f"Contract asset: {contract.asset}")
    print(f"Contract layer: {contract.layer}")

    # In your Dagster code, the asset name matches directly:
    #
    #   @asset(key="claims_silver")
    #   def claims_silver(context, database):
    #       ...
    #       database.write(df, target="silver.claims", context=context)
    #
    # The contract's asset field ("claims_silver") matches what
    # AssetKey.to_user_string() returns ("claims_silver").
    # --- cookbook:end ---

    assert contract is not None
    assert contract.asset == "claims_silver"


@pytest.mark.cookbook(
    title="Asset Naming: Multi-Component AssetKey with UI Hierarchy",
    description=(
        "Use multi-component ``AssetKey`` for Dagster UI grouping while keeping "
        "the contract ``asset`` field as the bare name. The contract loader "
        "automatically falls back to matching the last component of a "
        "slash-separated name."
    ),
    category="contracts",
)
def test_multi_component_key_resolution(tmp_path: Path) -> None:
    """Show how multi-component AssetKeys resolve to contracts."""
    # Contract uses the bare asset name -- no slashes
    (tmp_path / "fda_ndc_directory.contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "6d44c230-10c5-40a9-9d8c-7d0345a8afc3"
asset: fda_ndc_directory
layer: silver
schema:
  columns:
    - name: product_id
      type: string
      nullable: false
      pii: false
    - name: brand_name
      type: string
      nullable: true
      pii: false
"""
    )

    # --- cookbook:start ---
    from moncpipelib.contracts import load_contract_for_asset

    # In Dagster, you might define a multi-component key for UI grouping:
    #
    #   @asset(key=AssetKey(["reference_silver", "fda_ndc_directory"]))
    #   def fda_ndc_directory_silver(context, database):
    #       ...
    #
    # AssetKey.to_user_string() produces "reference_silver/fda_ndc_directory".
    # The loader tries an exact match first, then falls back to the last
    # component ("fda_ndc_directory") -- matching the contract.

    # Simulating what the IO manager does internally:
    dagster_name = "reference_silver/fda_ndc_directory"

    contract = load_contract_for_asset(dagster_name, layer="silver", search_paths=[tmp_path])
    print(f"Dagster name:   {dagster_name}")
    print(f"Contract asset: {contract.asset}")
    print("Resolved:       yes (last-component fallback)")
    # --- cookbook:end ---

    assert contract is not None
    assert contract.asset == "fda_ndc_directory"


@pytest.mark.cookbook(
    title="Asset Naming: Same Asset Across Layers",
    description=(
        "When the same logical asset exists at multiple layers (e.g., bronze and "
        "silver), each contract uses the same ``asset`` name but a different "
        "``layer``. The ``layer`` parameter disambiguates at lookup time."
    ),
    category="contracts",
)
def test_same_asset_across_layers(tmp_path: Path) -> None:
    """Show layer disambiguation for same-named assets."""
    (tmp_path / "claims_bronze.contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "550e8400-e29b-41d4-a716-446655440000"
asset: claims
layer: bronze
schema:
  columns:
    - name: claim_id
      type: string
      nullable: false
      pii: false
"""
    )
    (tmp_path / "claims_silver.contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "661e9500-f39c-52e5-b827-557766551111"
asset: claims
layer: silver
schema:
  columns:
    - name: claim_id
      type: string
      nullable: false
      pii: false
    - name: amount
      type: decimal
      nullable: false
      pii: false
"""
    )

    # --- cookbook:start ---
    from moncpipelib.contracts import load_contract_for_asset

    # Same asset name, different layers -- layer disambiguates
    bronze = load_contract_for_asset("claims", layer="bronze", search_paths=[tmp_path])
    silver = load_contract_for_asset("claims", layer="silver", search_paths=[tmp_path])

    print(f"Bronze: asset={bronze.asset}, layer={bronze.layer}, cols={len(bronze.schema.columns)}")
    print(f"Silver: asset={silver.asset}, layer={silver.layer}, cols={len(silver.schema.columns)}")

    # In Dagster, the layer is derived automatically from the target schema:
    #
    #   database.write(df, target="bronze.claims", context=context)  -> layer="bronze"
    #   database.write(df, target="silver.claims", context=context)  -> layer="silver"
    # --- cookbook:end ---

    assert bronze is not None and bronze.layer == "bronze"
    assert silver is not None and silver.layer == "silver"
    assert len(silver.schema.columns) > len(bronze.schema.columns)


@pytest.mark.cookbook(
    title="Asset Naming: Multi-Client Pipelines",
    description=(
        "When the same pipeline pattern is implemented per client, use "
        "client-prefixed asset names: ``{client}_{pipeline}``. This keeps names "
        "globally unique, avoids contract collisions, and works with "
        "``AssetKey`` hierarchy for clean UI grouping. Each client gets its "
        "own contract with potentially different columns, PII flags, or SLA."
    ),
    category="contracts",
)
def test_multi_client_naming(tmp_path: Path) -> None:
    """Show per-client asset naming with contracts."""
    (tmp_path / "cox_claims_load.contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "aaa11111-1111-1111-1111-111111111111"
asset: cox_claims_load
layer: silver
description: Claims ingestion for Cox
owner:
  team: data-engineering
schema:
  columns:
    - name: claim_id
      type: string
      nullable: false
      pii: false
    - name: member_id
      type: string
      nullable: false
      pii: true
    - name: plan_code
      type: string
      nullable: true
      pii: false
"""
    )
    (tmp_path / "aetna_claims_load.contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "bbb22222-2222-2222-2222-222222222222"
asset: aetna_claims_load
layer: silver
description: Claims ingestion for Aetna
owner:
  team: data-engineering
schema:
  columns:
    - name: claim_id
      type: string
      nullable: false
      pii: false
    - name: subscriber_id
      type: string
      nullable: false
      pii: true
    - name: network_tier
      type: string
      nullable: true
      pii: false
"""
    )

    # --- cookbook:start ---
    from moncpipelib.contracts import load_contract_for_asset

    # Convention: {client}_{pipeline} as the asset name
    # Each client has its own contract, pipeline_id, and potentially
    # different columns, PII flags, or SLA requirements.

    cox = load_contract_for_asset("cox_claims_load", search_paths=[tmp_path])
    aetna = load_contract_for_asset("aetna_claims_load", search_paths=[tmp_path])

    print(f"Cox asset:   {cox.asset}")
    print(f"Cox PII cols: {[c.name for c in cox.schema.columns if c.pii]}")
    print(f"Aetna asset: {aetna.asset}")
    print(f"Aetna PII cols: {[c.name for c in aetna.schema.columns if c.pii]}")

    # In Dagster, use multi-component keys for UI grouping:
    #
    #   @asset(key=AssetKey(["silver", "cox", "cox_claims_load"]))
    #   def cox_claims_load(context, database):
    #       database.write(df, target="silver.cox_claims_load", context=context)
    #
    #   @asset(key=AssetKey(["silver", "aetna", "aetna_claims_load"]))
    #   def aetna_claims_load(context, database):
    #       database.write(df, target="silver.aetna_claims_load", context=context)
    #
    # Each client's assets appear grouped in the Dagster UI:
    #   silver > cox > cox_claims_load
    #   silver > aetna > aetna_claims_load
    #
    # While each contract is globally unique by asset name alone.
    # --- cookbook:end ---

    assert cox is not None and cox.asset == "cox_claims_load"
    assert aetna is not None and aetna.asset == "aetna_claims_load"
    # Different clients can have different PII columns
    cox_pii = {c.name for c in cox.schema.columns if c.pii}
    aetna_pii = {c.name for c in aetna.schema.columns if c.pii}
    assert cox_pii == {"member_id"}
    assert aetna_pii == {"subscriber_id"}
