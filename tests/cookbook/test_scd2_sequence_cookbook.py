"""Cookbook: SCD2 Sequence Column (seq_id) Lifecycle.

Demonstrates per-business-key version sequencing through partition-scoped
writes and cross-partition reconciliation.
"""

from __future__ import annotations

import inspect

import pytest


@pytest.mark.cookbook(
    title="SCD2 Sequence Column: Per-Business-Key Version Numbering",
    description=(
        "The ``seq_id`` column assigns a monotonic integer (1, 2, 3, ...) to "
        "each version of a business entity. During partition-scoped writes, "
        "sequence numbers are scoped to the active partition via "
        "``MAX(seq_id) + 1``. After ``reconcile_scd2()`` stitches the "
        "cross-partition timeline, the renumbering step repairs gaps and "
        "duplicates so ``seq_id`` reflects position in the unified timeline."
    ),
    category="scd2",
)
def test_scd2_sequence_lifecycle() -> None:
    """Show how seq_id flows through writes and reconciliation."""
    # --- cookbook:start ---
    # 1. Schema: add seq_id as an INTEGER column alongside the SCD2 columns
    #
    #    CREATE TABLE silver.dim_product (
    #        id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    #        product_id   TEXT NOT NULL,
    #        product_name TEXT,
    #        region       TEXT,             -- partition column
    #        row_hash     TEXT,
    #        effective_from DATE,
    #        effective_to   DATE,
    #        is_current     BOOLEAN,
    #        seq_id         INTEGER          -- version sequence per business key
    #    );

    # 2. Partition-scoped writes assign seq_id per-partition
    #
    #    Within each partition, scd2_finalize() computes:
    #      COALESCE(MAX(seq_id) WHERE product_id = ? AND region IN (?), 0) + 1
    #
    #    After loading PROD-001 into US (seq_id=1) and EU (seq_id=1):
    #
    #    | product_id | region | seq_id | is_current | effective_from |
    #    |------------|--------|--------|------------|----------------|
    #    | PROD-001   | US     | 1      | true       | 2025-01-01     |
    #    | PROD-001   | EU     | 1      | true       | 2025-02-01     |
    #
    #    Both partitions independently start at seq_id=1 for the same BK.

    # 3. reconcile_scd2() stitches the timeline and renumbers seq_id
    #
    #    result = database.reconcile_scd2(
    #        target="silver.dim_product",
    #        business_key=["product_id"],
    #    )
    #
    #    After reconciliation:
    #
    #    | product_id | region | seq_id | is_current | effective_from | effective_to |
    #    |------------|--------|--------|------------|----------------|--------------|
    #    | PROD-001   | US     | 1      | false      | 2025-01-01     | 2025-02-01   |
    #    | PROD-001   | EU     | 2      | true       | 2025-02-01     | 9999-12-31   |
    #
    #    - Timeline stitched: US version's effective_to now points to EU's effective_from
    #    - seq_id renumbered: ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY effective_from)
    #    - result["rows_renumbered"] reports how many seq_id values were updated

    # 4. Opting out: set sequence_col=None in SCD2Config
    #
    #    from moncpipelib.config import SCD2Config
    #    database.reconcile_scd2(
    #        target="silver.dim_product",
    #        business_key=["product_id"],
    #        scd2=SCD2Config(sequence_col=None),  # skip renumbering
    #    )
    #
    #    Tables without a seq_id column are also silently skipped.

    # 5. Idempotency: running reconcile_scd2() again is safe
    #
    #    The renumber step uses IS DISTINCT FROM to skip rows that already
    #    have the correct seq_id, so result["rows_renumbered"] == 0 on
    #    repeated runs with no new data.
    # --- cookbook:end ---

    # Verify the API surface exists
    from moncpipelib.config import SCD2Config
    from moncpipelib.resources.postgres import PostgresResource

    sig = inspect.signature(PostgresResource.reconcile_scd2)
    assert "scd2" in sig.parameters

    cfg = SCD2Config()
    assert cfg.sequence_col == "seq_id"

    cfg_no_seq = SCD2Config(sequence_col=None)
    assert cfg_no_seq.sequence_col is None
    assert "seq_id" not in cfg_no_seq.managed_columns
