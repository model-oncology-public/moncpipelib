"""Cookbook tests for querying ``data_lineage.metadata``.

Each test here doubles as a documentation example.  Issue #334 added a
system-known JSONB payload on every contract-carrying write so dashboard
authors can slice the lineage table by write mode, partition shape, and
contract outcome without joining to ``contract_validation_runs``.

Note: ``from __future__ import annotations`` is intentionally omitted
here because Dagster's ``@asset`` decorator resolves type annotations
eagerly, and the PEP 563 stringification breaks resolution inside local
scopes.
"""

import pytest


@pytest.mark.cookbook(
    title="Querying data_lineage.metadata for write-mode and partition shape",
    description=(
        "Every write from ``PostgresResource.write()`` now lands a "
        "small JSONB payload on ``data_lineage.metadata``: "
        "``write_mode``, ``partition_column``, ``partition_keys``, "
        "``contract_enforcement``, and (single-path only) "
        "``contract_status``.  This makes it possible to slice the "
        "lineage table by operational shape without joining to "
        "``contract_validation_runs``.  Asymmetry to keep in mind: on "
        "the batched-write path, ``contract_status`` is absent and "
        "``write_mode`` / ``partition_column`` reflect caller input "
        "rather than the contract-reconciled value.  See issue #334."
    ),
    category="lineage",
)
def test_cookbook_query_data_lineage_metadata() -> None:
    """Document the typical ``metadata->>`` query shapes."""
    # --- cookbook:start ---
    # ``data_lineage.metadata`` is a queryable JSONB column populated on
    # every contract-carrying write.  The keys are:
    #
    #   write_mode             "full_refresh" / "upsert" / "append" / "scd2"
    #   partition_column       partition column name (when partitioned)
    #   partition_keys         active partition keys (capped at 50 + sentinel)
    #   contract_enforcement   "error" / "warn" / "silent"
    #   contract_status        "passed" / "failed" / "warned"  (single-path only)
    #
    # Counts of writes by mode in the last day:
    SQL_BY_MODE = """
    SELECT
        metadata->>'write_mode' AS write_mode,
        COUNT(*) AS writes
    FROM lineage.data_lineage
    WHERE processed_at >= now() - interval '1 day'
      AND metadata IS NOT NULL
    GROUP BY metadata->>'write_mode'
    ORDER BY writes DESC;
    """

    # Recent failed contract writes by asset:
    SQL_FAILED_CONTRACTS = """
    SELECT
        asset_name,
        layer,
        processed_at,
        metadata->>'write_mode' AS write_mode
    FROM lineage.data_lineage
    WHERE metadata->>'contract_status' = 'failed'
      AND processed_at >= now() - interval '7 days'
    ORDER BY processed_at DESC;
    """

    # Writes against a specific partition:
    SQL_BY_PARTITION = """
    SELECT lineage_id, asset_name, processed_at
    FROM lineage.data_lineage
    WHERE metadata->'partition_keys' ? '2026-05-15'
    ORDER BY processed_at DESC
    LIMIT 50;
    """

    # The payload carries no PHI -- only operational metadata (write
    # mode, partition keys, contract enforcement level).  Per-check
    # validation details (which rows failed, sample failures) continue
    # to live in ``lineage.contract_validation_runs``, which has its
    # own data classification.
    print("Three illustrative query shapes for data_lineage.metadata:")
    print("  1. Counts by write_mode in the last day")
    print("  2. Recent failed contract writes (single-path only)")
    print("  3. Writes against a specific partition key")
    # --- cookbook:end ---

    # Lightweight assertions: the SQL strings reference the documented
    # keys.  No DB roundtrip in cookbook tests.
    assert "write_mode" in SQL_BY_MODE
    assert "contract_status" in SQL_FAILED_CONTRACTS
    assert "partition_keys" in SQL_BY_PARTITION
