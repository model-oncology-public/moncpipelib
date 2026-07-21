"""Cookbook tests for PII metadata tracking via the column_metadata sidecar table.

Each test here doubles as a documentation example. The code between
``# --- cookbook:start ---`` and ``# --- cookbook:end ---`` markers is
extracted by the cookbook pytest plugin and rendered into docs/cookbook.md.
"""

from __future__ import annotations  # noqa: I001

import pytest


@pytest.mark.cookbook(
    title="Column Metadata Model (SCD2 PII Tracking)",
    description=(
        "moncpipelib tracks PII and PHI classification metadata in a dedicated "
        "``lineage.column_metadata`` SCD2 table. Each row records the "
        "classification tags for a column during a specific time period. When "
        "tags change, the current record is closed and a new one is opened, "
        "preserving full audit history for HIPAA/HITRUST compliance.\n\n"
        "The ``ColumnMetadata`` SQLAlchemy model is the source of truth for "
        "Alembic migrations. At runtime, ``PostgresResource`` syncs PII flags "
        "from data contracts into this table on every write."
    ),
    category="lineage",
)
def test_column_metadata_model() -> None:
    """Demonstrate the ColumnMetadata SQLAlchemy model."""
    # --- cookbook:start ---
    from moncpipelib.lineage import ColumnMetadata

    # Inspect the table structure
    table = ColumnMetadata.__table__
    print(f"Table: {table.schema}.{table.name}")
    print(f"Primary key: {[c.name for c in table.primary_key.columns]}")
    print("Columns:")
    for col in table.columns:
        nullable = "NULL" if col.nullable else "NOT NULL"
        print(f"  {col.name:20s} {str(col.type):25s} {nullable}")
    # --- cookbook:end ---

    assert table.schema == "lineage"
    assert table.name == "column_metadata"
    pk_names = [c.name for c in table.primary_key.columns]
    assert "schema_name" in pk_names
    assert "table_name" in pk_names
    assert "column_name" in pk_names
    assert "valid_from" in pk_names


@pytest.mark.cookbook(
    title="PII and PHI Tags from Data Contracts",
    description=(
        "Data contracts define ``pii: true/false`` and ``phi: true/false`` on "
        "each column. ``pii`` defaults to ``true`` for safety, and ``phi`` "
        "defaults to the ``pii`` value -- so a column is PHI-suspect unless "
        "an engineer affirmatively clears it with ``phi: false``. The two "
        "diverge for e.g. provider identifiers (PII but not PHI). At write "
        "time, ``PostgresResource`` reads these flags and syncs them as JSONB "
        "tags into ``lineage.column_metadata``. The JSONB approach is "
        "extensible -- future tags like ``sensitivity`` or ``retention`` can "
        "be added without any DDL changes."
    ),
    category="lineage",
)
def test_pii_tags_from_contract() -> None:
    """Show how PII / PHI flags translate to JSONB tags."""
    # --- cookbook:start ---
    import json

    from moncpipelib.contracts import Column, ColumnType, DataContract, Schema

    contract = DataContract(
        version="1.0",
        pipeline_id="550e8400-e29b-41d4-a716-446655440000",
        asset="staging.patients",
        layer="bronze",
        schema=Schema(
            columns=[
                Column(name="patient_id", type=ColumnType.STRING, nullable=False, pii=True),
                Column(name="ssn", type=ColumnType.STRING, nullable=True, pii=True),
                # PII but not PHI: identifies the provider, not the patient
                Column(
                    name="provider_npi",
                    type=ColumnType.STRING,
                    nullable=False,
                    pii=True,
                    phi=False,
                ),
                Column(name="claim_id", type=ColumnType.STRING, nullable=False, pii=False),
                Column(name="status", type=ColumnType.STRING, nullable=False, pii=False),
            ]
        ),
    )

    # Build tags dict per column (same logic as PostgresResource._sync_pii_metadata)
    for col in contract.get_non_managed_columns():
        tags = {"pii": col.pii, "phi": col.phi}
        print(f"  {col.name:20s} -> {json.dumps(tags)}")
    # --- cookbook:end ---

    pii_cols = contract.get_pii_column_names()
    non_pii_cols = contract.get_non_pii_column_names()
    assert "patient_id" in pii_cols
    assert "ssn" in pii_cols
    assert "provider_npi" in pii_cols
    assert "claim_id" in non_pii_cols
    assert "status" in non_pii_cols

    phi_cols = contract.get_phi_column_names()
    assert "patient_id" in phi_cols
    assert "ssn" in phi_cols
    assert "provider_npi" not in phi_cols  # cleared by phi: false
    assert "claim_id" not in phi_cols  # phi mirrors pii=false
