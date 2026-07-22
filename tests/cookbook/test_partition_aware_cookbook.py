"""Cookbook tests for partition-aware write mode configuration.

Each test here doubles as a documentation example. The code between
``# --- cookbook:start ---`` and ``# --- cookbook:end ---`` markers is
extracted by the cookbook pytest plugin and rendered into docs/cookbook.md.

Note: ``from __future__ import annotations`` is intentionally omitted here
because Dagster's ``@asset`` decorator resolves type annotations eagerly,
and the PEP 563 stringification breaks resolution inside local scopes.
"""

import pytest

# ---------------------------------------------------------------------------
# Summary of all four partition-aware write modes
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Partition-Aware Write Modes: Complete Reference",
    description=(
        "When a Dagster asset is partitioned and ``partition_column`` is set, "
        "moncpipelib scopes all write operations to the active partition(s). "
        "Each write mode handles partitions differently. This reference shows "
        "the SQL behavior and requirements for all four modes."
    ),
    category="write_modes",
)
def test_partition_modes_summary() -> None:
    """Demonstrate all four write modes with partition scoping."""
    # --- cookbook:start ---
    # All four write modes support partitioned assets. The behavior and
    # requirements differ per mode:
    #
    # +-----------------+----------------------------------------+------------------------+
    # | Write Mode      | Partition Behavior                     | Guard Rail              |
    # +-----------------+----------------------------------------+------------------------+
    # | full_refresh    | DELETE WHERE partition_column IN (...)  | partition_column        |
    # |                 | then INSERT. Other partitions untouched.| REQUIRED                |
    # +-----------------+----------------------------------------+------------------------+
    # | upsert          | Standard upsert. partition_column must | partition_column must   |
    # |                 | be in primary_key to prevent cross-     | be in primary_key       |
    # |                 | partition matching.                     |                         |
    # +-----------------+----------------------------------------+------------------------+
    # | scd2            | Change detection scoped to active       | partition_column        |
    # |                 | partition. detect_deletes also scoped.  | REQUIRED. detect_deletes|
    # |                 |                                         | needs partition_column. |
    # +-----------------+----------------------------------------+------------------------+
    # | append          | No special handling. Additive-only,     | None                    |
    # |                 | always safe with or without partitions. |                         |
    # +-----------------+----------------------------------------+------------------------+

    # --- Configuration ---
    # partition_column is set via contract sink or asset metadata:
    #
    # Contract YAML:
    #   sinks:
    #     - type: table
    #       mode: full_refresh
    #       partition_column: event_date
    #
    # Or asset metadata:
    #   @asset(metadata={"partition_column": "event_date"})

    # --- SQL patterns per mode ---

    print("=== full_refresh (partitioned) ===")
    print("  DELETE FROM silver.events WHERE \"event_date\" IN ('2025-01-15')")
    print("  INSERT INTO silver.events (...) VALUES (...)")
    print("  -> Only replaces rows for the active partition date")
    print()

    print("=== upsert (partitioned) ===")
    print("  INSERT INTO silver.visits (...) VALUES (...)")
    print('  ON CONFLICT ("patient_id", "visit_date") DO UPDATE ...')
    print("  -> partition_column (visit_date) MUST be in primary_key")
    print("  -> Prevents matching rows across partitions")
    print()

    print("=== scd2 (partitioned) ===")
    print("  -- Change detection scoped:")
    print("  SELECT ... FROM staging s")
    print('  LEFT JOIN silver.products t ON t."product_id" = s."product_id"')
    print('    AND t."is_current" = true')
    print("    AND t.\"load_period\" IN ('2025-H1')")
    print("  -- Expirations and inserts also scoped to partition")
    print("  -> detect_deletes only expires within the active partition")
    print()

    print("=== append (partitioned) ===")
    print("  INSERT INTO bronze.logs (...) VALUES (...)")
    print("  -> No partition scoping needed (additive-only)")
    print("  -> partition_column is optional")
    # --- cookbook:end ---

    # Basic assertion to satisfy pytest
    assert True


# ---------------------------------------------------------------------------
# Partition-aware full refresh
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Partition-Aware Full Refresh",
    description=(
        "When a Dagster asset is partitioned and ``partition_column`` is set in "
        "metadata, the IO manager automatically scopes ``full_refresh`` writes to "
        "only the active partition. Instead of deleting all rows, it executes "
        "``DELETE WHERE partition_column IN (...)`` for the active partition keys, "
        "then inserts the new data. Other partitions remain untouched."
    ),
    category="write_modes",
)
def test_partition_aware_full_refresh() -> None:
    """Demonstrate partition-scoped full refresh via asset metadata."""
    # --- cookbook:start ---
    import polars as pl
    from dagster import (
        AssetExecutionContext,
        DailyPartitionsDefinition,
        Definitions,
        asset,
    )

    from moncpipelib.io_managers import PostgresIOManager
    from moncpipelib.resources import PostgresResource

    database = PostgresResource(
        host="db.example.com",
        user="writer",
        password="secret",
        database="analytics",
    )

    io_manager = PostgresIOManager(
        postgres_resource=database,
        default_schema="silver",
    )

    # Define a daily-partitioned asset with partition_column.
    # The IO manager reads the Dagster partition context automatically.
    @asset(
        io_manager_key="silver_io_manager",
        partitions_def=DailyPartitionsDefinition(start_date="2025-01-01"),
        metadata={
            "write_mode": "full_refresh",
            "partition_column": "event_date",
        },
    )
    def daily_events(_context: AssetExecutionContext) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "event_id": ["E001", "E002"],
                "event_date": ["2025-01-15", "2025-01-15"],
                "event_type": ["admission", "discharge"],
            }
        )

    defs = Definitions(
        assets=[daily_events],
        resources={"silver_io_manager": io_manager},
    )

    # When materialized for partition "2025-01-15", the IO manager executes:
    #   DELETE FROM silver.daily_events WHERE "event_date" IN ('2025-01-15')
    #   INSERT INTO silver.daily_events ...
    # Rows for other dates are never touched.

    print("Partition-aware full refresh:")
    print('  write_mode: "full_refresh"')
    print('  partition_column: "event_date"')
    print()
    print("SQL pattern for single partition materialization:")
    print('  DELETE FROM silver.daily_events WHERE "event_date" IN (%s)')
    print("  INSERT INTO silver.daily_events ...")
    print()
    print("For multi-partition backfills, all specified dates are scoped:")
    print('  DELETE ... WHERE "event_date" IN (%s, %s, ...)')
    # --- cookbook:end ---

    asset_node = defs.get_assets_def("daily_events")
    specs = list(asset_node.specs)
    assert specs[0].metadata["write_mode"] == "full_refresh"
    assert specs[0].metadata["partition_column"] == "event_date"


# ---------------------------------------------------------------------------
# Partition-aware upsert with composite key
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Partition-Aware Upsert with Composite Key",
    description=(
        "When using ``upsert`` mode with partitioned assets, the ``partition_column`` "
        "must be included in the ``primary_key`` to prevent cross-partition conflicts. "
        "The IO manager enforces this as a guard rail -- if the partition column is "
        "missing from the primary key, a ``ContractViolationError`` is raised before "
        "any data is written."
    ),
    category="write_modes",
)
def test_partition_aware_upsert_composite_key() -> None:
    """Demonstrate partition-aware upsert requiring composite primary key."""
    # --- cookbook:start ---
    import polars as pl
    from dagster import (
        AssetExecutionContext,
        DailyPartitionsDefinition,
        Definitions,
        asset,
    )

    from moncpipelib.io_managers import PostgresIOManager
    from moncpipelib.resources import PostgresResource

    database = PostgresResource(
        host="db.example.com",
        user="writer",
        password="secret",
        database="analytics",
    )

    io_manager = PostgresIOManager(
        postgres_resource=database,
        default_schema="silver",
    )

    # For partitioned upserts, include partition_column in primary_key.
    # This prevents upsert from matching rows across partitions.
    @asset(
        io_manager_key="silver_io_manager",
        partitions_def=DailyPartitionsDefinition(start_date="2025-01-01"),
        metadata={
            "write_mode": "upsert",
            "primary_key": ["patient_id", "visit_date"],  # includes partition_column
            "partition_column": "visit_date",
        },
    )
    def patient_visits(_context: AssetExecutionContext) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "patient_id": ["PAT-001", "PAT-002"],
                "visit_date": ["2025-01-15", "2025-01-15"],
                "diagnosis": ["flu", "checkup"],
            }
        )

    defs = Definitions(
        assets=[patient_visits],
        resources={"silver_io_manager": io_manager},
    )

    # The guard rail ensures safe partition isolation:
    #   primary_key: ["patient_id", "visit_date"]  -- OK (includes partition_column)
    #   primary_key: ["patient_id"]                 -- ERROR: would match across dates

    print("Partition-aware upsert:")
    print('  write_mode: "upsert"')
    print('  primary_key: ["patient_id", "visit_date"]')
    print('  partition_column: "visit_date"')
    print()
    print("Guard rail: partition_column MUST be in primary_key for upsert.")
    print("Without it, upsert could update rows from other partitions.")
    # --- cookbook:end ---

    asset_node = defs.get_assets_def("patient_visits")
    specs = list(asset_node.specs)
    assert specs[0].metadata["write_mode"] == "upsert"
    assert "visit_date" in specs[0].metadata["primary_key"]


# ---------------------------------------------------------------------------
# Partition-scoped SCD2
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Partition-Scoped SCD2",
    description=(
        "SCD2 change detection can be scoped to a partition by setting "
        "``partition_column`` in metadata. When active, the IO manager only compares "
        "incoming data against existing records in the same partition -- records from "
        "other partitions are never expired or compared. This is critical for "
        "partitioned pipelines where each run only processes a slice of the data."
    ),
    category="write_modes",
)
def test_partition_scoped_scd2() -> None:
    """Demonstrate SCD2 with partition-scoped change detection."""
    # --- cookbook:start ---
    import polars as pl
    from dagster import (
        AssetExecutionContext,
        DailyPartitionsDefinition,
        Definitions,
        asset,
    )

    from moncpipelib.io_managers import PostgresIOManager
    from moncpipelib.resources import PostgresResource

    database = PostgresResource(
        host="db.example.com",
        user="writer",
        password="secret",
        database="analytics",
    )

    io_manager = PostgresIOManager(
        postgres_resource=database,
        default_schema="silver",
    )

    # SCD2 with partition scoping: change detection is limited to
    # the active partition. Records from other partitions are untouched.
    @asset(
        io_manager_key="silver_io_manager",
        partitions_def=DailyPartitionsDefinition(start_date="2025-01-01"),
        metadata={
            "write_mode": "scd2",
            "business_key": ["patient_id"],
            "partition_column": "report_date",
        },
    )
    def patient_status_scd2(_context: AssetExecutionContext) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "patient_id": ["PAT-001", "PAT-002"],
                "report_date": ["2025-01-15", "2025-01-15"],
                "status": ["active", "discharged"],
            }
        )

    defs = Definitions(
        assets=[patient_status_scd2],
        resources={"silver_io_manager": io_manager},
    )

    # When materialized for partition "2025-01-15":
    # - Only compares against rows WHERE "report_date" IN ('2025-01-15')
    # - Expires changed rows only within that partition
    # - Inserts new versions only for that partition's data
    # - Records from other dates are completely untouched

    print("Partition-scoped SCD2:")
    print('  write_mode: "scd2"')
    print('  business_key: ["patient_id"]')
    print('  partition_column: "report_date"')
    print()
    print("Change detection SQL is scoped:")
    print('  LEFT JOIN ... ON bk AND t."report_date" IN (%s)')
    print("  Only records within the partition are compared/expired.")
    # --- cookbook:end ---

    asset_node = defs.get_assets_def("patient_status_scd2")
    specs = list(asset_node.specs)
    assert specs[0].metadata["write_mode"] == "scd2"
    assert specs[0].metadata["partition_column"] == "report_date"


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Partition Guard Rails",
    description=(
        "The IO manager enforces guard rails to prevent unsafe partition + write mode "
        "combinations. These fire as ``ContractViolationError`` before any data is "
        "written, ensuring misconfigurations are caught early."
    ),
    category="write_modes",
)
def test_partition_guard_rails() -> None:
    """Demonstrate the three partition safety guard rails."""
    # --- cookbook:start ---
    from moncpipelib.contracts.exceptions import ContractViolationError

    # The IO manager validates these rules when context.has_partition_key is True:

    # Guard 1: full_refresh or scd2 + partitioned context + no partition_column
    # ERROR: "no partition_column configured"
    # Without partition_column, full_refresh would DELETE all rows (not just
    # the active partition), and SCD2 would compare against the entire table.
    # Fix: add partition_column to metadata or contract.

    # Guard 2: upsert + partition_column NOT in primary_key
    # ERROR: "primary_key does not include partition_column"
    # Upsert ON CONFLICT matches on primary_key. If the partition column is
    # not in the PK, a row from partition A could match and update a row
    # from partition B.
    # Fix: include partition_column in primary_key.

    # Guard 3: scd2 + detect_deletes + no partition_column
    # ERROR: "detect_deletes would expire records from all partitions"
    # detect_deletes expires records absent from the incoming data. Without
    # partition scoping, records from other partitions would be expired.
    # Fix: add partition_column to metadata or contract.

    # Note: append mode does NOT require partition_column (it is additive-only).

    print("Partition guard rails (ContractViolationError):")
    print()
    print("1. full_refresh/scd2 + partitioned + no partition_column")
    print('   -> "no partition_column configured"')
    print()
    print("2. upsert + partition_column not in primary_key")
    print('   -> "primary_key does not include partition_column"')
    print()
    print("3. scd2 + detect_deletes + no partition_column")
    print('   -> "detect_deletes would expire records from all partitions"')
    print()
    print("4. append + partitioned (no guard -- additive-only, always safe)")
    # --- cookbook:end ---

    # Verify the exception class exists and is importable
    assert issubclass(ContractViolationError, Exception)
