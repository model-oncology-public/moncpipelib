"""Cookbook tests for contract-based Dagster asset check generation.

Each test here doubles as a documentation example. The code between
``# --- cookbook:start ---`` and ``# --- cookbook:end ---`` markers is
extracted by the cookbook pytest plugin and rendered into docs/cookbook.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Cookbook examples
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Generate Asset Checks from a Data Contract",
    description=(
        "Build individual Dagster asset checks from a DataContract object. "
        "Each column test and table expectation becomes a separate "
        "AssetChecksDefinition, giving granular pass/fail visibility in the "
        "Dagster UI. A df_loader callback supplies the DataFrame at check "
        "execution time."
    ),
    category="contracts",
)
def test_generate_checks_from_contract() -> None:
    """Demonstrate generating Dagster asset checks from a contract."""
    # --- cookbook:start ---
    import polars as pl
    from dagster import AssetCheckExecutionContext

    from moncpipelib.contracts import (
        Column,
        ColumnTest,
        ColumnType,
        DataContract,
        Schema,
        Severity,
        TableExpectation,
        generate_asset_checks_from_contract,
    )

    # Define a contract with schema, column tests, and table expectations
    contract = DataContract(
        version="1.0",
        pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        asset="orders_bronze",
        layer="bronze",
        schema=Schema(
            columns=[
                Column(
                    name="order_id",
                    type=ColumnType.STRING,
                    nullable=False,
                    tests=[
                        ColumnTest(test_type="not_null"),
                        ColumnTest(test_type="unique"),
                    ],
                ),
                Column(
                    name="amount",
                    type=ColumnType.DECIMAL,
                    nullable=False,
                    tests=[
                        ColumnTest(
                            test_type="greater_than",
                            parameters={"value": 0},
                            severity=Severity.WARN,
                        ),
                    ],
                ),
            ],
        ),
        expectations=[
            TableExpectation(
                expectation_type="row_count",
                parameters={"min": 1, "max": 10000},
            ),
        ],
    )

    # The df_loader is called at check execution time to fetch data
    def load_orders(_context: AssetCheckExecutionContext) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "order_id": ["ORD-001", "ORD-002"],
                "amount": [99.99, 149.50],
            }
        )

    # Generate checks -- batched=True (default) bundles all rules into one op
    checks = generate_asset_checks_from_contract(
        contract,
        "orders_bronze",  # asset key (string, list, or AssetKey)
        load_orders,
    )

    # One definition containing all check specs (runs in a single pod)
    print(f"Definitions: {len(checks)}")
    total_specs = sum(len(chk.check_keys) for chk in checks)
    print(f"Total check specs: {total_specs}")
    # 1 schema check + 2 order_id tests + 1 amount test + 1 row_count = 5
    for chk in checks:
        for key in sorted(chk.check_keys, key=lambda k: k.name):
            print(f"  - {key.name}")
    # --- cookbook:end ---

    assert len(checks) == 1  # single batched definition
    assert total_specs == 5


@pytest.mark.cookbook(
    title="Discover Contract Checks from a Directory",
    description=(
        "Scan a directory for ``*.contract.yaml`` files and generate Dagster "
        "asset checks for every contract found. Provide a ``df_loader_factory`` "
        "that maps each asset name to a callable returning its DataFrame. "
        "Optionally pass ``asset_key_prefix`` to namespace the generated checks."
    ),
    category="contracts",
)
def test_discover_checks_from_directory(tmp_path: Path) -> None:
    """Demonstrate discovering contracts and generating checks."""
    # Write sample contract files to the temp directory
    (tmp_path / "orders.contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: orders
layer: bronze
schema:
  columns:
    - name: order_id
      type: string
      nullable: false
      tests:
        - not_null
    - name: total
      type: decimal
      nullable: false
"""
    )
    (tmp_path / "patients.contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "b2c3d4e5-f6a7-8901-bcde-f12345678901"
asset: patients
layer: bronze
schema:
  columns:
    - name: patient_id
      type: string
      nullable: false
      tests:
        - not_null
        - unique
"""
    )

    # --- cookbook:start ---
    import polars as pl
    from dagster import AssetCheckExecutionContext

    from moncpipelib.contracts import discover_contract_checks

    # Factory that returns a df_loader for each discovered asset
    def make_loader(asset_name: str):
        sample_data = {
            "orders": pl.DataFrame({"order_id": ["O-1"], "total": [50.0]}),
            "patients": pl.DataFrame({"patient_id": ["P-1"]}),
        }

        def loader(_context: AssetCheckExecutionContext) -> pl.DataFrame:
            return sample_data[asset_name]

        return loader

    # Discover all *.contract.yaml files and generate checks
    checks = discover_contract_checks(
        tmp_path,  # directory to scan
        make_loader,
        asset_key_prefix=["bronze"],  # optional: prefix asset keys
    )

    print(f"Checks discovered: {len(checks)}")
    for chk in checks:
        for key in sorted(chk.check_keys, key=lambda k: k.name):
            print(f"  - {key.asset_key.path} / {key.name}")
    # --- cookbook:end ---

    # 2 contracts -> 2 batched definitions (one per contract)
    # orders: 1 schema + 1 not_null = 2 specs
    # patients: 1 schema + 2 tests = 3 specs
    assert len(checks) == 2
    total_specs = sum(len(chk.check_keys) for chk in checks)
    assert total_specs == 5


@pytest.mark.cookbook(
    title="Generate Contract Checks from the IO Manager",
    description=(
        "The recommended way to generate contract checks is via "
        "``PostgresIOManager.make_contract_checks()``.  It recursively scans "
        "for ``*.contract.yaml`` files, reads each contract's sink ``schema`` "
        "for per-contract routing, and defers database connections to check "
        "execution time (compatible with Dagster ``EnvVar``)."
    ),
    category="contracts",
)
def test_make_contract_checks_io_manager(tmp_path: Path) -> None:
    """Demonstrate IO manager contract checks with per-contract schema."""
    # Write contracts with different sink schemas
    (tmp_path / "silver").mkdir()
    (tmp_path / "silver" / "claims.contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "c3d4e5f6-a7b8-9012-cdef-123456789012"
asset: claims
layer: silver
sinks:
  - type: table
    schema: synthetic_silver
    table: claims
schema:
  columns:
    - name: claim_id
      type: string
      nullable: false
      pii: false
"""
    )
    (tmp_path / "gold").mkdir()
    (tmp_path / "gold" / "dim_patient.contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "d4e5f6a7-b890-1234-cdef-234567890123"
asset: dim_patient
layer: gold
sinks:
  - type: table
    schema: synthetic_gold
    table: dim_patient
schema:
  columns:
    - name: patient_id
      type: string
      nullable: false
      pii: false
"""
    )

    # --- cookbook:start ---
    from moncpipelib.io_managers import PostgresIOManager
    from moncpipelib.resources import PostgresResource

    # Create the resource with connection config -- credentials live in one place.
    database = PostgresResource(
        host="db.example.com",
        user="reader",
        password="secret",  # use EnvVar("DB_PASSWORD") in production
        database="analytics",
    )

    # One IO manager, pointed at the root contracts directory.
    # Works with EnvVar -- connection is deferred to check execution time.
    pg_io = PostgresIOManager(
        postgres_resource=database,
        default_schema="silver",
    )

    # Recursively discovers contracts in subdirectories.
    # Each contract's sink schema drives routing -- no single db_schema needed.
    checks = pg_io.make_contract_checks(
        tmp_path,  # e.g., "defs/" in a load_from_defs_folder layout
    )

    print(f"Checks generated: {len(checks)}")
    all_keys = sorted(
        (key.name for chk in checks for key in chk.check_keys),
    )
    for name in all_keys:
        print(f"  - {name}")
    print()
    print("Each contract reads from its own sink schema:")
    print("  claims     -> synthetic_silver.claims")
    print("  dim_patient -> synthetic_gold.dim_patient")
    print()
    print("Pass to Dagster Definitions:")
    print("  defs = Definitions(assets=[...], asset_checks=checks)")
    # --- cookbook:end ---

    # 1 schema check per contract = 2
    assert len(checks) == 2


@pytest.mark.cookbook(
    title="Standalone Contract Checks with connection_factory",
    description=(
        "Use standalone ``make_contract_checks()`` with a ``connection_factory`` "
        "when you don't have an IO Manager instance.  The factory defers credential "
        "resolution to check execution time, avoiding Dagster ``EnvVar`` issues.  "
        "Each contract's sink ``schema`` drives per-contract routing."
    ),
    category="contracts",
)
def test_make_contract_checks_standalone(tmp_path: Path) -> None:
    """Demonstrate standalone make_contract_checks with connection_factory."""
    # Write a sample contract with sink schema
    (tmp_path / "claims.contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "c3d4e5f6-a7b8-9012-cdef-123456789012"
asset: claims
layer: bronze
sinks:
  - type: table
    schema: bronze
    table: claims
schema:
  columns:
    - name: claim_id
      type: string
      nullable: false
      pii: false
"""
    )

    # --- cookbook:start ---
    import psycopg

    from moncpipelib.contracts.checks import make_contract_checks

    # connection_factory defers credential resolution to execution time.
    # This avoids EnvVar issues when called from definitions.py.
    checks = make_contract_checks(
        tmp_path,  # directory with *.contract.yaml files (recursive)
        connection_factory=lambda: psycopg.connect(
            host="db.example.com",
            user="reader",
            password="secret",  # use os.environ["DB_PASSWORD"] in production
            dbname="analytics",
        ),
        # Schema comes from each contract's sink -- no db_schema needed.
        # Fallback if a contract has no sink schema:
        default_schema="bronze",
    )

    print(f"Checks generated: {len(checks)}")
    for chk in checks:
        for key in chk.check_keys:
            print(f"  - {key.name}")

    # Pass to Dagster Definitions:
    # defs = Definitions(assets=[...], asset_checks=checks)
    # --- cookbook:end ---

    assert len(checks) == 1  # 1 schema check for the single contract


@pytest.mark.cookbook(
    title="SCD2 Sinks: Contract Checks Validate Current Rows",
    description=(
        "Contracts whose sink declares ``mode: scd2`` have their asset "
        "checks automatically scoped to current rows (``is_current = "
        "TRUE``). An SCD2 table legitimately repeats business keys across "
        "history rows, so an unscoped ``unique`` check would fail on the "
        "first change wave (issue #418). With scoping, ``unique: true`` on "
        "a business key means 'unique within the current snapshot' on both "
        "validation surfaces: the write path validates the incoming frame "
        "pre-merge, and asset checks validate current rows post-write."
    ),
    category="contracts",
)
def test_scd2_checks_scope_to_current_rows() -> None:
    """Demonstrate SCD2 current-row scoping for contract checks."""
    # --- cookbook:start ---
    import polars as pl

    from moncpipelib.contracts.validators import run_column_test

    # An SCD2 table after one change wave: BLA 017016 changed, so it has
    # an expired history row AND a current row -- the key repeats.
    scd2_table = pl.DataFrame(
        {
            "bla_number": ["017016", "017016", "125554"],
            "applicant": ["Genentech", "Genentech Inc", "Amgen"],
            "is_current": [False, True, True],
        }
    )

    # Unscoped (full history): unique on the business key fails --
    # this is why unscoped full-table checks broke on SCD2 sinks.
    full_history = run_column_test(
        df=scd2_table, column="bla_number", test_type="unique", parameters={}
    )
    print(f"Full history unique: passed={full_history.passed}")

    # Current rows only: what checks generated for a ``mode: scd2`` sink
    # validate automatically (no contract changes needed).
    current_only = run_column_test(
        df=scd2_table.filter(pl.col("is_current")),
        column="bla_number",
        test_type="unique",
        parameters={},
    )
    print(f"Current rows unique: passed={current_only.passed}")

    # SQL-pushdown checks apply the same scoping in-database:
    #   SELECT COUNT(DISTINCT "bla_number"), COUNT(*)
    #   FROM reference_gold.fda_purplebook_bla
    #   WHERE "is_current" = TRUE
    # Check result metadata records the scope:
    #   scope: current rows only (is_current = TRUE)
    # --- cookbook:end ---

    assert not full_history.passed
    assert current_only.passed
