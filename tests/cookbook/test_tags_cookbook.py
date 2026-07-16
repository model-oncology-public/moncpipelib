"""Cookbook tests for the tags module.

Each test here doubles as a documentation example. The code between
``# --- cookbook:start ---`` and ``# --- cookbook:end ---`` markers is
extracted by the cookbook pytest plugin and rendered into docs/cookbook.md.
"""

from __future__ import annotations

import pytest

from moncpipelib.contracts.models import (
    SLA,
    Column,
    ColumnType,
    DataContract,
    LineageConfig,
    Owner,
    Schema,
)


def _sample_contract() -> DataContract:
    """Build a realistic contract for cookbook examples."""
    return DataContract(
        version="1.0",
        pipeline_id="550e8400-e29b-41d4-a716-446655440000",
        asset="claims_bronze",
        layer="bronze",
        schema=Schema(
            columns=[
                Column(
                    name="claim_id",
                    type=ColumnType.STRING,
                    nullable=False,
                    primary_key=True,
                    pii=False,
                ),
                Column(
                    name="patient_id",
                    type=ColumnType.STRING,
                    nullable=False,
                    pii=True,
                ),
            ]
        ),
        owner=Owner(team="data-engineering"),
        sla=SLA(freshness_hours=24, update_frequency="daily"),
        lineage=LineageConfig(source_system="sftp"),
        tags={"team/priority": "high"},
    )


@pytest.mark.cookbook(
    title="Building Tags from a Contract",
    description=(
        "Use ContractTags.from_contract() to auto-derive Dagster-compatible "
        "tags from a data contract. The tags include layer, owner, pipeline_id, "
        "SLA/PII flags, and any user-defined tags from the contract YAML."
    ),
    category="tags",
)
def test_contract_tags_from_contract() -> None:
    """Demonstrate building tags from a single contract."""
    contract = _sample_contract()

    # --- cookbook:start ---
    from moncpipelib.tags import ContractTags

    # Build tags from a contract -- auto-derives safe structural metadata
    tags = ContractTags.from_contract(contract)

    # Convert to a plain dict for Dagster
    tag_dict = tags.to_dict()
    print(f"Layer:         {tag_dict['moncpipelib/layer']}")
    print(f"Owner:         {tag_dict['moncpipelib/owner']}")
    print(f"Has SLA:       {tag_dict['moncpipelib/has_sla']}")
    print(f"Has PII:       {tag_dict['moncpipelib/has_pii']}")
    print(f"Source system: {tag_dict['moncpipelib/source_system']}")
    print(f"User tag:      {tag_dict['team/priority']}")
    # --- cookbook:end ---

    assert tag_dict["moncpipelib/layer"] == "bronze"
    assert tag_dict["moncpipelib/owner"] == "data-engineering"
    assert tag_dict["moncpipelib/has_sla"] == "true"
    assert tag_dict["moncpipelib/has_pii"] == "true"
    assert tag_dict["team/priority"] == "high"


@pytest.mark.cookbook(
    title="Composing Runtime Tags with RunTags",
    description=(
        "Use RunTags to compose contract-derived tags with runtime information "
        "like image versions or environment labels. RunTags is a mutable builder "
        "with a fluent API -- all add methods return self for chaining."
    ),
    category="tags",
)
def test_run_tags_composition() -> None:
    """Demonstrate composing tags from multiple sources."""
    contract = _sample_contract()

    # --- cookbook:start ---
    from moncpipelib.tags import ContractTags, RunTags

    # Start with contract-derived tags
    contract_tags = ContractTags.from_contract(contract)

    # Compose with runtime tags using the builder
    IMAGE_TAG = {"image/version": "v2.4.1", "image/registry": "acr.io/pipelines"}

    run_tags = RunTags()
    run_tags.add_contract_tags(contract_tags)
    run_tags.add_tags(IMAGE_TAG)
    run_tags.add_tag("env", "production")

    # Use with Dagster job definitions
    tag_dict = run_tags.to_dict()
    print(f"Total tags:    {len(run_tags)}")
    print(f"Layer:         {tag_dict['moncpipelib/layer']}")
    print(f"Image version: {tag_dict['image/version']}")
    print(f"Environment:   {tag_dict['env']}")

    # Fluent API also works
    quick_tags = RunTags().add_contract_tags(contract_tags).add_tag("env", "staging").to_dict()
    print(f"Quick env:     {quick_tags['env']}")
    # --- cookbook:end ---

    assert len(run_tags) >= 8
    assert tag_dict["image/version"] == "v2.4.1"
    assert tag_dict["env"] == "production"
    assert quick_tags["env"] == "staging"
