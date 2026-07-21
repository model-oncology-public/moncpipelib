"""Cookbook tests for migrating to the universal PostgresIOManager pattern.

Each test here doubles as a documentation example. The code between
``# --- cookbook:start ---`` and ``# --- cookbook:end ---`` markers is
extracted by the cookbook pytest plugin and rendered into docs/cookbook.md.

Note: ``from __future__ import annotations`` is intentionally omitted here
because Dagster's ``@asset`` decorator resolves type annotations eagerly,
and the PEP 563 stringification breaks resolution inside local scopes.
"""

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Before / After: definitions.py migration
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Migrate from Per-Layer IO Managers to a Single Universal IO Manager",
    description=(
        "Instead of creating one ``PostgresIOManager`` per schema (bronze, silver, gold), "
        "create a single instance with ``default_schema`` and let each asset declare its "
        "target schema via metadata or contract sinks.  Connection config is specified once, "
        "and ``db_schema`` / ``layer`` / ``table_suffix_to_strip`` are no longer needed."
    ),
    category="universal_io_manager",
)
def test_definitions_migration() -> None:
    """Show the before/after for definitions.py IO manager setup."""
    # --- cookbook:start ---
    import polars as pl
    from dagster import AssetExecutionContext, Definitions, asset

    from moncpipelib.io_managers import PostgresIOManager
    from moncpipelib.resources import PostgresResource

    # ---- BEFORE: one IO manager per layer, repeated connection config ----
    # silver_io_manager = PostgresIOManager(
    #     host="db.example.com", user="writer", password="secret",
    #     database="analytics",
    #     db_schema="silver",             # deprecated
    #     layer="silver",                  # deprecated
    #     table_suffix_to_strip="_silver", # deprecated
    # )
    # gold_io_manager = PostgresIOManager(
    #     host="db.example.com", user="writer", password="secret",
    #     database="analytics",
    #     db_schema="gold",
    #     layer="gold",
    #     table_suffix_to_strip="_gold",
    # )

    # ---- AFTER: one resource + one IO manager, per-asset schema routing ----
    database = PostgresResource(
        host="db.example.com",
        user="writer",
        password="secret",
        database="analytics",
    )

    pg_io = PostgresIOManager(
        postgres_resource=database,
        default_schema="silver",  # fallback when asset doesn't specify
    )

    # Silver assets: target_schema in metadata (or omit to use default_schema)
    @asset(
        io_manager_key="pg_io",
        metadata={"target_schema": "silver"},
    )
    def patient_claims(_context: AssetExecutionContext) -> pl.DataFrame:
        return pl.DataFrame({"claim_id": ["C-001"], "amount": [100.0]})

    # Gold assets: same IO manager, different target_schema
    @asset(
        io_manager_key="pg_io",
        metadata={
            "target_schema": "gold",
            "write_mode": "upsert",
            "primary_key": ["patient_id"],
        },
    )
    def dim_patient(_context: AssetExecutionContext) -> pl.DataFrame:
        return pl.DataFrame({"patient_id": ["P-001"], "name": ["Alice"]})

    defs = Definitions(
        assets=[patient_claims, dim_patient],
        resources={"pg_io": pg_io},
    )

    print("Before: 3 IO managers x 6 connection args each = 18 redundant lines")
    print("After:  1 IO manager, connection config specified once")
    print()
    print("Schema routing per asset:")
    print("  patient_claims -> silver (via target_schema metadata)")
    print("  dim_patient    -> gold   (via target_schema metadata)")
    # --- cookbook:end ---

    # Verify definitions were created
    assert defs.get_assets_def("patient_claims") is not None
    assert defs.get_assets_def("dim_patient") is not None


# ---------------------------------------------------------------------------
# Contract-driven schema routing (no target_schema in metadata needed)
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Contract-Driven Schema Routing",
    description=(
        "When a data contract declares a sink with ``schema``, ``table``, and ``mode``, "
        "the IO manager reads all routing and write config from the contract.  The asset "
        "definition only needs ``io_manager_key`` -- no schema, write mode, or key "
        "configuration in metadata.  This is the recommended approach for "
        "production pipelines."
    ),
    category="universal_io_manager",
)
def test_contract_driven_routing(tmp_path: Path) -> None:
    """Show how contracts eliminate metadata boilerplate on assets."""
    # --- cookbook:start ---
    from pathlib import Path

    from moncpipelib.contracts import load_contract

    # Data contract drives all write configuration:
    contract_yaml = """\
version: "1.0"
pipeline_id: "550e8400-e29b-41d4-a716-446655440000"
asset: fda_ndc_directory
layer: silver
description: FDA NDC directory with SCD2 versioning

sinks:
  - type: table
    schema: reference_silver          # IO manager routes writes here
    table: fda_ndc_directory          # target table name
    mode: scd2                        # write mode
    business_key: [product_id]        # SCD2 business key
    tracked_columns: [brand_name, generic_name, dosage_form]
    detect_deletes: true

schema:
  columns:
    - name: product_id
      type: string
      nullable: false
    - name: brand_name
      type: string
      nullable: true
    - name: generic_name
      type: string
      nullable: true
    - name: dosage_form
      type: string
      nullable: true
"""

    # Write the contract to a file (in production, colocated with asset code)
    contract_file = Path(tmp_path) / "fda_ndc_directory.contract.yaml"
    contract_file.write_text(contract_yaml)

    contract = load_contract(contract_file)

    # The asset definition becomes minimal:
    # @asset(io_manager_key="pg_io")     # that's it!
    # def fda_ndc_directory(...): ...
    #
    # The IO manager finds the colocated contract and uses:
    #   schema:          reference_silver (from sink)
    #   write_mode:      scd2 (from sink.mode)
    #   business_key:    [product_id] (from sink)
    #   tracked_columns: [brand_name, generic_name, dosage_form] (from sink)
    #   detect_deletes:  true (from sink)

    print("Contract-driven routing eliminates metadata boilerplate:")
    print()
    print("  @asset(io_manager_key='pg_io')")
    print("  def fda_ndc_directory(context) -> pl.DataFrame: ...")
    print()
    print("All write configuration comes from the contract sink:")
    sinks = contract.sinks
    assert sinks is not None
    for sink in sinks:
        print(f"  schema:          {sink.get('schema')}")
        print(f"  table:           {sink.get('table')}")
        print(f"  mode:            {sink.get('mode')}")
        print(f"  business_key:    {sink.get('business_key')}")
        print(f"  detect_deletes:  {sink.get('detect_deletes')}")
    # --- cookbook:end ---

    assert contract.asset == "fda_ndc_directory"
    assert len(contract.sinks) == 1  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Asset naming: drop the _silver/_gold suffix
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Asset Naming and Table Resolution",
    description=(
        "The IO manager resolves the target table name via a priority cascade: "
        "(1) contract sink ``table`` field, (2) asset name.  When a contract "
        "exists with a matching sink, the ``table`` field wins -- the asset can "
        "be named anything.  Contracts with exactly one table sink always match, "
        "even if the asset name differs (e.g. ``fda_ndc_directory_silver`` matches "
        "a sink with ``table: fda_ndc_directory``).  Without a contract, the asset "
        "name *is* the table name, so dropping ``_silver`` / ``_gold`` suffixes is "
        "recommended to avoid needing ``table_suffix_to_strip``."
    ),
    category="universal_io_manager",
)
def test_asset_naming_migration() -> None:
    """Show table name resolution and the naming convention change."""
    # --- cookbook:start ---
    import polars as pl
    from dagster import AssetExecutionContext, Definitions, asset

    from moncpipelib.io_managers import PostgresIOManager
    from moncpipelib.resources import PostgresResource

    database = PostgresResource(
        host="db.example.com",
        user="writer",
        password="secret",
        database="analytics",
    )

    pg_io = PostgresIOManager(
        postgres_resource=database,
        default_schema="silver",
    )

    # ---- BEFORE: asset named with layer suffix ----
    # @asset(io_manager_key="silver_io_manager")
    # def patient_claims_silver(...) -> pl.DataFrame: ...
    #
    # IO manager strips "_silver", writes to: silver.patient_claims
    # Required: table_suffix_to_strip="_silver" on the IO manager

    # ---- AFTER (no contract): name the asset to match the table ----
    @asset(
        io_manager_key="pg_io",
        metadata={"target_schema": "silver"},
    )
    def patient_claims(_context: AssetExecutionContext) -> pl.DataFrame:
        return pl.DataFrame({"claim_id": ["C-001"], "amount": [100.0]})

    # Asset name "patient_claims" -> writes to silver.patient_claims
    # No suffix stripping needed.

    # ---- AFTER (with contract): asset name doesn't matter ----
    # When a contract has a single table sink, the IO manager always
    # matches it -- even if the asset name differs from the sink's
    # `table` field. The sink's `table` becomes the physical table name.
    #
    # Example: asset "fda_ndc_directory_silver" with contract sink:
    #   table: fda_ndc_directory
    #   schema: reference_silver
    # -> writes to reference_silver.fda_ndc_directory
    #
    # This allows incremental migration without renaming every asset.

    defs = Definitions(
        assets=[patient_claims],
        resources={"pg_io": pg_io},
    )

    print("Table name resolution (priority cascade):")
    print("  1. Contract sink 'table' field (if contract exists)")
    print("  2. Asset name (context.asset_key.path[-1])")
    print()
    print("Without a contract:")
    print("  asset 'patient_claims' -> table 'patient_claims'")
    print("  Recommendation: name assets to match their target table")
    print()
    print("With a contract (single table sink):")
    print("  The IO manager always matches the lone table sink,")
    print("  regardless of asset name. The sink's table field wins.")
    print("  e.g. asset 'fda_ndc_directory_silver' + sink table 'fda_ndc_directory'")
    print("       -> writes to reference_silver.fda_ndc_directory")
    print()
    print("Schema resolution (priority cascade):")
    print("  1. schema_override (integration test isolation)")
    print("  2. Contract sink 'schema' field")
    print("  3. target_schema from asset metadata")
    print("  4. default_schema from IO manager constructor")
    # --- cookbook:end ---

    assert defs.get_assets_def("patient_claims") is not None


# ---------------------------------------------------------------------------
# for_testing() factory for integration tests
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Create Test IO Managers with for_testing()",
    description=(
        "Use ``for_testing()`` to create test-isolated IO manager clones.  All original "
        "configuration (connection, write behavior, OpenLineage, enforcement mode) is "
        "automatically preserved.  Only test-specific overrides (schema, table prefix, "
        "contract paths) are applied.  This replaces the fragile pattern of manually "
        "copying IO manager fields in test harnesses."
    ),
    category="universal_io_manager",
)
def test_for_testing_factory() -> None:
    """Show how for_testing() simplifies integration test setup."""
    # --- cookbook:start ---
    from moncpipelib.contracts import ContractEnforcementMode
    from moncpipelib.io_managers import PostgresIOManager
    from moncpipelib.io_managers.postgres import WriteMode
    from moncpipelib.resources import PostgresResource

    # Production resource with connection + feature config
    database = PostgresResource(
        host="db.example.com",
        user="writer",
        password="secret",
        database="analytics",
        openlineage_url="https://lineage.internal/api/v1",
        openlineage_namespace="analytics-prod",
    )

    # Production IO manager delegates to the resource
    pg_io = PostgresIOManager(
        postgres_resource=database,
        default_schema="silver",
        enforce_contracts=ContractEnforcementMode.ERROR,
    )

    # Create a test-isolated clone -- one line instead of 15+
    test_io = pg_io.for_testing(
        test_schema="integration_tests",
        table_prefix="ci_abc123_",
        contract_search_paths=["pipelines/silver/claims/"],
    )

    # Verify: test overrides applied
    assert test_io.schema_override == "integration_tests"
    assert test_io.table_prefix == "ci_abc123_"
    assert test_io.contract_search_paths == ["pipelines/silver/claims/"]

    # Verify: production config automatically preserved
    assert test_io.postgres_resource.host == "db.example.com"
    assert test_io.postgres_resource.database == "analytics"
    assert test_io.enforce_contracts == ContractEnforcementMode.ERROR
    assert test_io.postgres_resource.openlineage_url == "https://lineage.internal/api/v1"

    # Verify: original not mutated
    assert pg_io.schema_override is None
    assert pg_io.table_prefix is None

    # ---- BEFORE: manual field-by-field reconstruction ----
    # original_io = definitions.silver_io_manager
    # test_io = PostgresIOManager(
    #     postgres_resource=original_io.postgres_resource,
    #     db_schema=original_io.db_schema,        # must remember each field
    #     layer=original_io.layer,                  # miss one = silent bug
    #     table_suffix_to_strip=original_io.table_suffix_to_strip,
    #     schema_override="integration_tests",
    #     table_prefix="ci_abc123_",
    # )

    print("for_testing() preserves all production config automatically:")
    print(f"  host:               {test_io.postgres_resource.host}")
    print(f"  database:           {test_io.postgres_resource.database}")
    print(f"  enforce_contracts:  {test_io.enforce_contracts.value}")
    print(f"  openlineage_url:    {test_io.postgres_resource.openlineage_url}")
    print()
    print("Test-specific overrides:")
    print(f"  schema_override:    {test_io.schema_override}")
    print(f"  table_prefix:       {test_io.table_prefix}")
    print()
    print("Any field can be overridden via **kwargs:")

    custom = pg_io.for_testing(
        test_schema="integration_tests",
        write_mode=WriteMode.FULL_REFRESH,  # override write mode for test
    )
    print(f"  write_mode:         {custom.write_mode.value}")
    # --- cookbook:end ---


# ---------------------------------------------------------------------------
# Complete migration example: before/after side by side
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Complete Migration: Before and After",
    description=(
        "End-to-end comparison showing a pipeline definition migrated from the legacy "
        "per-layer IO manager pattern to the universal pattern.  Shows ``definitions.py`` "
        "changes, asset definition changes, and how the contract drives all write "
        "configuration."
    ),
    category="universal_io_manager",
)
def test_complete_migration_example() -> None:
    """Show a complete pipeline migration from legacy to universal pattern."""
    # --- cookbook:start ---
    import polars as pl
    from dagster import AssetExecutionContext, Definitions, asset

    from moncpipelib.io_managers import PostgresIOManager
    from moncpipelib.resources import PostgresResource

    # ========================================================================
    # definitions.py -- BEFORE
    # ========================================================================
    # silver_io_manager = PostgresIOManager(
    #     host=EnvVar("DB_HOST"), port=EnvVar.int("DB_PORT"),
    #     user=EnvVar("DB_USER"), password=EnvVar("DB_PASSWORD"),
    #     database=EnvVar("DB_NAME"),
    #     db_schema="silver",
    #     layer="silver",
    #     table_suffix_to_strip="_silver",
    # )
    # gold_io_manager = PostgresIOManager(
    #     host=EnvVar("DB_HOST"), ...same 5 args...,
    #     db_schema="gold",
    #     layer="gold",
    #     table_suffix_to_strip="_gold",
    # )
    # reference_silver_io_manager = PostgresIOManager(
    #     host=EnvVar("DB_HOST"), ...same 5 args...,
    #     db_schema="reference_silver",
    #     layer="silver",
    #     table_suffix_to_strip="_silver",
    # )
    # resources = {
    #     "silver_io_manager": silver_io_manager,
    #     "gold_io_manager": gold_io_manager,
    #     "reference_silver_io_manager": reference_silver_io_manager,
    # }

    # ========================================================================
    # definitions.py -- AFTER
    # ========================================================================
    database = PostgresResource(
        host="db.example.com",
        user="writer",
        password="secret",
        database="analytics",
    )

    pg_io = PostgresIOManager(
        postgres_resource=database,
        default_schema="silver",  # most assets write to silver
    )

    # ========================================================================
    # Asset definition -- BEFORE
    # ========================================================================
    # @graph_asset
    # def fda_ndc_directory_silver():
    #     """Write to reference_silver.fda_ndc_directory."""
    #     # op uses io_manager_key="reference_silver_io_manager"
    #     # IO manager strips "_silver" -> table "fda_ndc_directory"
    #     # Schema comes from db_schema="reference_silver" on the IO manager
    #     ...

    # ========================================================================
    # Asset definition -- AFTER
    # ========================================================================
    @asset(
        io_manager_key="pg_io",
        # Contract colocated at fda_ndc_directory.contract.yaml handles:
        #   schema: reference_silver
        #   table: fda_ndc_directory
        #   mode: scd2
        #   business_key: [product_id]
        #   tracked_columns: [brand_name, generic_name, ...]
        #   detect_deletes: true
    )
    def fda_ndc_directory(_context: AssetExecutionContext) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "product_id": ["PROD-001"],
                "brand_name": ["Aspirin"],
                "generic_name": ["acetylsalicylic acid"],
                "dosage_form": ["TABLET"],
            }
        )

    @asset(
        io_manager_key="pg_io",
        metadata={
            "target_schema": "gold",
            "write_mode": "upsert",
            "primary_key": ["patient_id"],
        },
    )
    def dim_patient(_context: AssetExecutionContext) -> pl.DataFrame:
        return pl.DataFrame({"patient_id": ["P-001"], "name": ["Alice"]})

    defs = Definitions(
        assets=[fda_ndc_directory, dim_patient],
        resources={"pg_io": pg_io},
    )

    print("Migration summary:")
    print()
    print("definitions.py:")
    print("  Before: 3 IO managers x 8 args each = ~24 lines of config")
    print("  After:  1 IO manager  x 5 args       = ~6 lines of config")
    print()
    print("Asset definitions:")
    print("  Before: io_manager_key='reference_silver_io_manager'")
    print("          asset name 'fda_ndc_directory_silver' (suffix stripped)")
    print("  After:  io_manager_key='pg_io'")
    print("          asset name 'fda_ndc_directory' (matches table directly)")
    print("          contract drives schema + write mode + SCD2 config")
    print()
    print("What the IO manager resolves at write time:")
    print("  fda_ndc_directory -> reference_silver.fda_ndc_directory (from contract)")
    print("  dim_patient       -> gold.dim_patient (from target_schema metadata)")
    # --- cookbook:end ---

    assert defs.get_assets_def("fda_ndc_directory") is not None
    assert defs.get_assets_def("dim_patient") is not None


# ---------------------------------------------------------------------------
# Production definitions.py with defs/ layout
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Production definitions.py with defs/ Layout",
    description=(
        "A complete ``definitions.py`` example using Dagster's "
        "``load_from_defs_folder()`` pattern with a single universal IO manager.  "
        "Contracts are colocated with asset code under ``defs/`` and auto-discovered "
        "recursively.  Contract checks are generated via "
        "``io_mgr.make_contract_checks()``.  Assets with legacy layer suffixes "
        "(e.g. ``fda_ndc_directory_silver``) work correctly because the contract "
        "sink's ``table`` field drives physical table resolution."
    ),
    category="universal_io_manager",
)
def test_production_defs_layout(tmp_path: Path) -> None:
    """Show a realistic definitions.py with defs/ layout and contract checks."""
    # Set up a realistic defs/ directory with colocated contracts
    silver_dir = tmp_path / "defs" / "silver" / "reference" / "fda_ndc_directory"
    silver_dir.mkdir(parents=True)
    (silver_dir / "fda_ndc_directory_silver.contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "550e8400-e29b-41d4-a716-446655440000"
asset: fda_ndc_directory_silver
layer: silver
sinks:
  - type: table
    schema: reference_silver
    table: fda_ndc_directory
    mode: scd2
    business_key: [product_id]
    tracked_columns: [brand_name, generic_name, dosage_form]
    detect_deletes: true
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
    - name: generic_name
      type: string
      nullable: true
      pii: false
    - name: dosage_form
      type: string
      nullable: true
      pii: false
"""
    )

    gold_dir = tmp_path / "defs" / "gold" / "dim_patient"
    gold_dir.mkdir(parents=True)
    (gold_dir / "dim_patient_gold.contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "660e8400-e29b-41d4-a716-446655440001"
asset: dim_patient_gold
layer: gold
sinks:
  - type: table
    schema: synthetic_gold
    table: dim_patient
    mode: upsert
schema:
  columns:
    - name: patient_id
      type: string
      nullable: false
      pii: false
"""
    )

    # --- cookbook:start ---
    from pathlib import Path

    from moncpipelib.io_managers import PostgresIOManager
    from moncpipelib.resources import PostgresResource

    # ----------------------------------------------------------------
    # definitions.py -- single resource + universal IO manager
    # ----------------------------------------------------------------
    database = PostgresResource(
        host="db.example.com",  # use EnvVar("DB_HOST") in production
        user="writer",
        password="secret",  # use EnvVar("DB_PASSWORD") in production
        database="analytics",
    )

    pg_io = PostgresIOManager(
        postgres_resource=database,
        default_schema="silver",  # fallback when no contract sink schema
    )

    # ----------------------------------------------------------------
    # Contract checks -- auto-discovered from the defs/ directory tree
    # ----------------------------------------------------------------
    defs_path = Path(tmp_path) / "defs"  # e.g., Path(__file__).parent / "defs"

    # Recursively finds all *.contract.yaml files in defs/ subdirectories.
    # Each contract's sink schema drives routing (no single db_schema needed).
    # Connection is deferred to check execution time (EnvVar compatible).
    checks = pg_io.make_contract_checks(defs_path)

    # ----------------------------------------------------------------
    # Wire it all together
    # ----------------------------------------------------------------
    # defs = Definitions.merge(
    #     Definitions(
    #         resources={"pg_io": pg_io},
    #         asset_checks=checks,
    #     ),
    #     load_from_defs_folder(
    #         project_root=Path(__file__).parent.parent.parent,
    #     ),
    # )

    print("Production definitions.py with defs/ layout:")
    print()
    print("Directory structure:")
    print("  defs/")
    print("    silver/reference/fda_ndc_directory/")
    print("      fda_ndc_directory_silver.contract.yaml")
    print("      __init__.py  # contains @asset def")
    print("    gold/dim_patient/")
    print("      dim_patient_gold.contract.yaml")
    print("      __init__.py  # contains @asset def")
    print()
    print(f"Contract checks discovered: {len(checks)}")
    for chk in checks:
        for key in chk.check_keys:
            print(f"  - {key.name}")
    print()
    print("Key points:")
    print("  - One IO manager, connection config specified once")
    print("  - Contracts colocated with asset code in defs/ tree")
    print("  - make_contract_checks() recursively scans defs/")
    print("  - Each contract's sink schema drives per-contract routing")
    print("  - Legacy asset names (with _silver/_gold) work via sink table override")
    # --- cookbook:end ---

    # 1 schema check per contract = 2
    assert len(checks) == 2


# ---------------------------------------------------------------------------
# Legacy asset names with contract sink table override
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Legacy Asset Names with Contract Sink Table Override",
    description=(
        "During migration from per-layer IO managers, assets may retain layer "
        "suffixes (e.g. ``fda_ndc_directory_silver``).  The contract sink's "
        "``table`` field overrides the asset-derived table name, so the IO manager "
        "writes to the correct physical table (``fda_ndc_directory``) regardless "
        "of the asset name.  When a contract has exactly one table sink, the IO "
        "manager always matches it -- even if the asset name differs from the "
        "sink's ``table`` field.  This allows incremental migration without "
        "renaming every asset upfront."
    ),
    category="universal_io_manager",
)
def test_legacy_asset_name_sink_override(tmp_path: Path) -> None:
    """Show how contract sink table field overrides asset-derived table name."""
    (tmp_path / "fda_ndc_directory_silver.contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "550e8400-e29b-41d4-a716-446655440000"
asset: fda_ndc_directory_silver
layer: silver
sinks:
  - type: table
    schema: reference_silver
    table: fda_ndc_directory
    mode: scd2
    business_key: [product_id]
schema:
  columns:
    - name: product_id
      type: string
      nullable: false
"""
    )

    # --- cookbook:start ---
    from moncpipelib.contracts import load_contract

    # Contract for an asset with a legacy layer suffix
    contract = load_contract(tmp_path / "fda_ndc_directory_silver.contract.yaml")

    # The asset name has a layer suffix that doesn't match the sink table:
    #   asset name:  fda_ndc_directory_silver
    #   sink table:  fda_ndc_directory
    #
    # The IO manager resolves this automatically:
    #
    # 1. find_matching_sink: the contract has one table sink, so it always
    #    matches -- even though "fda_ndc_directory_silver" != "fda_ndc_directory"
    #
    # 2. _resolve_target: the matched sink's `table` field becomes the
    #    physical table name, and `schema` drives schema routing
    #
    # Result: writes to reference_silver.fda_ndc_directory (correct!)
    # Without the sink override, it would try reference_silver.fda_ndc_directory_silver

    assert contract.asset == "fda_ndc_directory_silver"
    assert contract.sinks is not None
    sink = contract.sinks[0]
    assert sink["table"] == "fda_ndc_directory"
    assert sink["schema"] == "reference_silver"

    print("Legacy asset name with contract sink table override:")
    print()
    print("  Asset name:           fda_ndc_directory_silver")
    print(f"  Contract sink table:  {sink['table']}")
    print(f"  Contract sink schema: {sink['schema']}")
    print()
    print("  IO manager resolution:")
    print("    1. Matches the single table sink (lenient fallback)")
    print(f"    2. Uses sink table '{sink['table']}' as physical table name")
    print(f"    3. Uses sink schema '{sink['schema']}' for routing")
    print(f"    4. Writes to: {sink['schema']}.{sink['table']}")
    print()
    print("  This allows incremental migration:")
    print("    - Keep legacy asset names (fda_ndc_directory_silver)")
    print("    - Contract sink drives correct table + schema resolution")
    print("    - No table_suffix_to_strip needed on the IO manager")
    print("    - Rename assets at your own pace (optional)")
    # --- cookbook:end ---

    assert sink["table"] == "fda_ndc_directory"
    assert sink["schema"] == "reference_silver"


# ---------------------------------------------------------------------------
# Resource-first pattern (recommended for new pipelines)
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Resource-First Pattern (Recommended)",
    description=(
        "For new pipelines, prefer ``PostgresResource.write()`` over IO managers. "
        "This pattern uses ``@asset`` with ``deps`` instead of ``@graph_asset`` + "
        "``@op``, and returns a ``MaterializeResult`` with write statistics. The "
        "asset explicitly owns its write target, eliminating implicit routing. "
        "Existing IO manager assets continue to work alongside resource-based assets."
    ),
    category="universal_io_manager",
)
def test_resource_first_pattern() -> None:
    """Show the recommended resource-first pattern for new pipelines."""
    # --- cookbook:start ---
    import polars as pl
    from dagster import AssetExecutionContext, Definitions, MaterializeResult, asset

    from moncpipelib import PostgresResource
    from moncpipelib.io_managers import PostgresIOManager

    # ---- IO Manager pattern (legacy, still supported) ----
    # The IO manager implicitly handles writes when the asset returns a DataFrame.
    # Schema routing is driven by metadata or contracts.
    # The IO manager shares the same PostgresResource for connection config.
    pg_io_resource = PostgresResource(
        host="db.example.com",
        user="writer",
        password="secret",
        database="analytics",
    )

    pg_io = PostgresIOManager(
        postgres_resource=pg_io_resource,
        default_schema="silver",
    )

    @asset(
        io_manager_key="pg_io",
        metadata={"target_schema": "silver"},
    )
    def patient_claims_legacy(_context: AssetExecutionContext) -> pl.DataFrame:
        return pl.DataFrame({"claim_id": ["C-001"], "amount": [100.0]})

    # ---- Resource pattern (recommended for new pipelines) ----
    # The asset explicitly writes using database.write() and returns metadata.
    # No implicit routing -- the target table is always explicit.
    database = PostgresResource(
        host="db.example.com",
        user="writer",
        password="secret",
        database="analytics",
        contract_search_paths=["defs/"],
    )

    @asset(deps=["provider_staging"])
    def dim_provider(
        context: AssetExecutionContext,
        database: PostgresResource,
    ) -> MaterializeResult:
        df = pl.DataFrame({"provider_id": ["NPI-001"], "name": ["Dr. Smith"]})
        result = database.write(df, target="gold.dim_provider", context=context)
        return MaterializeResult(metadata=result.to_dagster_metadata())

    defs = Definitions(
        assets=[patient_claims_legacy, dim_provider],
        resources={
            "pg_io": pg_io,
            "database": database,
        },
    )

    print("Pattern comparison:")
    print()
    print("IO Manager (legacy):")
    print("  @asset(io_manager_key='pg_io', metadata={'target_schema': 'silver'})")
    print("  def my_asset(context) -> pl.DataFrame:")
    print("      return df  # IO manager handles write implicitly")
    print()
    print("Resource (recommended for new pipelines):")
    print("  @asset(deps=['upstream'])")
    print("  def my_asset(context, database: PostgresResource) -> MaterializeResult:")
    print("      result = database.write(df, target='schema.table', context=context)")
    print("      return MaterializeResult(metadata=result.to_dagster_metadata())")
    print()
    print("Advantages of the resource pattern:")
    print("  - Explicit write target (no implicit routing)")
    print("  - No @graph_asset / @op boilerplate")
    print("  - Single resource for reads AND writes")
    print("  - WriteResult gives programmatic access to stats")
    print("  - Compatible with k8s_job_executor without env var issues")
    print("  - Both patterns coexist in the same Definitions")
    # --- cookbook:end ---

    assert defs.get_assets_def("patient_claims_legacy") is not None
    assert defs.get_assets_def("dim_provider") is not None
