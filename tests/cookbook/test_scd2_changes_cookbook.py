"""Cookbook examples for standalone SCD2 change detection."""

from __future__ import annotations

import polars as pl
import pytest


@pytest.mark.cookbook(
    title="SCD2 Change Detection: Standalone detect_changes",
    description=(
        "``detect_changes`` performs Python-side (Polars) SCD2 change detection "
        "between an incoming batch and the current dimension state. It is "
        "*informational*: the atomic write is still handled by the IO manager's "
        "Postgres-side CTE, but ``detect_changes`` gives a pipeline pre-write "
        "visibility into what would change -- useful for logging summaries, "
        "custom filtering, or feeding monitoring. Both DataFrames must carry a "
        "hash column (add it with ``compute_row_hash``). The returned "
        "``SCD2ChangeResult`` categorizes rows into ``new_records`` (key absent "
        "from current), ``changed_records`` (key present, hash differs), and "
        "``unchanged_records`` (key present, hash matches), plus a ``summary`` "
        "dict of counts."
    ),
    category="scd2",
)
def test_detect_changes_basic() -> None:
    """Demonstrate categorizing an incoming batch against current state."""
    # --- cookbook:start ---
    from moncpipelib import compute_row_hash, detect_changes

    hash_cols = ["product_id", "name", "price"]

    # Current dimension state (is_current=True rows already in the warehouse)
    current = pl.DataFrame(
        {
            "product_id": [1, 2, 4],
            "name": ["Widget", "Gadget", "Sprocket"],
            "price": ["9.99", "19.99", "4.99"],
        }
    ).with_columns(compute_row_hash(hash_cols))

    # Incoming batch: id 1 unchanged, id 2 reprices, id 3 is brand new
    incoming = pl.DataFrame(
        {
            "product_id": [1, 2, 3],
            "name": ["Widget", "Gadget", "Cog"],
            "price": ["9.99", "24.99", "14.99"],
        }
    ).with_columns(compute_row_hash(hash_cols))

    result = detect_changes(incoming, current, business_key="product_id")

    print("=== Summary ===")
    print(result.summary)
    print()
    print("New records:", result.new_records["product_id"].to_list())
    print("Changed records:", result.changed_records["product_id"].to_list())
    print("Unchanged records:", result.unchanged_records["product_id"].to_list())
    # --- cookbook:end ---

    assert result.summary == {
        "new": 1,
        "changed": 1,
        "unchanged": 1,
        "deleted": 0,
        "total_incoming": 3,
    }
    # Use set comparisons: Polars join result order is not guaranteed
    assert set(result.new_records["product_id"].to_list()) == {3}
    assert set(result.changed_records["product_id"].to_list()) == {2}
    assert set(result.unchanged_records["product_id"].to_list()) == {1}
    # Deletes not requested -> empty frame
    assert result.deleted_keys.height == 0


@pytest.mark.cookbook(
    title="SCD2 Change Detection: Composite Keys and Delete Detection",
    description=(
        "``detect_changes`` accepts a composite ``business_key`` (a list of "
        "columns) and, with ``detect_deletes=True``, also reports keys present "
        "in the current state but missing from the incoming batch via "
        "``deleted_keys``. This supports soft-delete or expire-on-absence "
        "workflows where rows that vanish upstream must be closed out. Deletes "
        "are off by default because many incremental feeds are partial and "
        "absence does not imply deletion."
    ),
    category="scd2",
)
def test_detect_changes_composite_and_deletes() -> None:
    """Demonstrate composite keys and opt-in delete detection."""
    # --- cookbook:start ---
    from moncpipelib import compute_row_hash, detect_changes

    hash_cols = ["region", "sku", "qty"]

    current = pl.DataFrame(
        {
            "region": ["US", "US", "EU"],
            "sku": ["A1", "B2", "A1"],
            "qty": ["10", "5", "7"],
        }
    ).with_columns(compute_row_hash(hash_cols))

    # (EU, A1) dropped out of the feed; (US, B2) restocked; (US, C3) is new
    incoming = pl.DataFrame(
        {
            "region": ["US", "US", "US"],
            "sku": ["A1", "B2", "C3"],
            "qty": ["10", "8", "3"],
        }
    ).with_columns(compute_row_hash(hash_cols))

    result = detect_changes(
        incoming,
        current,
        business_key=["region", "sku"],  # composite key
        detect_deletes=True,
    )

    print("=== Summary ===")
    print(result.summary)
    print()
    print("Deleted keys (in current, absent from incoming):")
    print(result.deleted_keys)
    # --- cookbook:end ---

    assert result.summary["new"] == 1  # (US, C3)
    assert result.summary["changed"] == 1  # (US, B2): 5 -> 8
    assert result.summary["unchanged"] == 1  # (US, A1)
    assert result.summary["deleted"] == 1  # (EU, A1)
    deleted = {
        (r, s)
        for r, s in zip(
            result.deleted_keys["region"].to_list(),
            result.deleted_keys["sku"].to_list(),
            strict=True,
        )
    }
    assert deleted == {("EU", "A1")}
