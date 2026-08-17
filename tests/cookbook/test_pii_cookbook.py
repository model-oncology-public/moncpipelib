"""Cookbook tests for PII column tracking and PII-aware rendering.

Each test here doubles as a documentation example. The code between
``# --- cookbook:start ---`` and ``# --- cookbook:end ---`` markers is
extracted by the cookbook pytest plugin and rendered into docs/cookbook.md.
"""

from __future__ import annotations  # noqa: I001

import pytest


# ---------------------------------------------------------------------------
# Cookbook examples
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="PII-Aware DataFrame Rendering",
    description=(
        "Use ``polars_to_md()`` to render a Polars DataFrame as a markdown "
        "table with automatic PII masking. When a data contract is provided, "
        "columns with ``pii: true`` (the default for unannotated columns) are "
        "replaced with a mask value. This prevents accidental PII exposure in "
        "logs and Dagster UI."
    ),
    category="rendering",
)
def test_pii_aware_rendering() -> None:
    """Demonstrate PII-masked DataFrame rendering with polars_to_md."""
    # --- cookbook:start ---
    import polars as pl

    from moncpipelib.contracts import Column, ColumnType, DataContract, Schema
    from moncpipelib.rendering import polars_to_md

    # Define a contract with PII annotations
    contract = DataContract(
        version="1.0",
        pipeline_id="550e8400-e29b-41d4-a716-446655440000",
        asset="patients",
        layer="bronze",
        schema=Schema(
            columns=[
                Column(name="patient_id", type=ColumnType.STRING, nullable=False, pii=True),
                Column(name="name", type=ColumnType.STRING, nullable=False, pii=True),
                Column(name="diagnosis_code", type=ColumnType.STRING, nullable=True, pii=False),
                Column(name="status", type=ColumnType.STRING, nullable=False, pii=False),
            ]
        ),
    )

    # Sample DataFrame
    df = pl.DataFrame(
        {
            "patient_id": ["PAT-001", "PAT-002", "PAT-003"],
            "name": ["Alice Smith", "Bob Jones", "Carol White"],
            "diagnosis_code": ["C50.1", "D05.9", None],
            "status": ["active", "discharged", "active"],
        }
    )

    # Render with PII masking -- PII columns are replaced with ***
    md = polars_to_md(df, contract=contract)
    print("With contract (PII masked):")
    print(md)

    # Render without contract -- no masking, caller is responsible
    md_raw = polars_to_md(df)
    print("\nWithout contract (no masking):")
    print(md_raw)

    # You can also pass explicit PII columns without a contract
    md_explicit = polars_to_md(df, pii_columns=["patient_id", "name"])
    print("\nWith explicit pii_columns list:")
    print(md_explicit)
    # --- cookbook:end ---

    # Verify PII columns are masked
    assert "PAT-001" not in md
    assert "Alice Smith" not in md
    assert "***" in md
    # Non-PII columns are visible
    assert "C50.1" in md
    assert "active" in md


@pytest.mark.cookbook(
    title="PII Column Tracking in Data Contracts",
    description=(
        "Data contracts support a ``pii`` boolean field on each column. "
        "Columns default to ``pii: true`` (safe by default) -- if a column "
        "is not explicitly annotated with ``pii: false``, it is treated as PII. "
        "Use ``get_pii_column_names()`` and ``get_non_pii_column_names()`` to "
        "query PII status programmatically."
    ),
    category="rendering",
)
def test_pii_column_tracking() -> None:
    """Demonstrate PII field usage in data contracts."""
    # --- cookbook:start ---
    from moncpipelib.contracts import Column, ColumnType, DataContract, Schema

    # Build a contract -- note pii defaults to True
    contract = DataContract(
        version="1.0",
        pipeline_id="550e8400-e29b-41d4-a716-446655440000",
        asset="claims",
        layer="bronze",
        schema=Schema(
            columns=[
                # Explicitly PII
                Column(name="patient_id", type=ColumnType.STRING, nullable=False, pii=True),
                Column(name="ssn", type=ColumnType.STRING, nullable=True, pii=True),
                # Explicitly NOT PII
                Column(name="claim_id", type=ColumnType.STRING, nullable=False, pii=False),
                Column(name="status", type=ColumnType.STRING, nullable=False, pii=False),
                # Unannotated -- defaults to pii=True (safe default)
                Column(name="provider_name", type=ColumnType.STRING, nullable=True),
            ]
        ),
    )

    pii_names = contract.get_pii_column_names()
    non_pii_names = contract.get_non_pii_column_names()

    print(f"PII columns:     {pii_names}")
    print(f"Non-PII columns: {non_pii_names}")
    print(f"Total columns:   {len(contract.schema.columns)}")

    # The safe default means unannotated columns are treated as PII
    provider_col = contract.get_column("provider_name")
    print(f"\nprovider_name.pii = {provider_col.pii}  (default -- safe)")
    print(f"claim_id.pii      = {contract.get_column('claim_id').pii}  (explicit opt-out)")
    # --- cookbook:end ---

    assert pii_names == ["patient_id", "ssn", "provider_name"]
    assert non_pii_names == ["claim_id", "status"]
    assert provider_col.pii is True
