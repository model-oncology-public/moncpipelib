"""Cookbook tests for the PostgresResource.write() pattern.

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
# Direct write with PostgresResource
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Direct Write with PostgresResource",
    description=(
        "Write a DataFrame to PostgreSQL using the resource directly, bypassing IO "
        "managers. This is the recommended pattern for new pipelines. The asset declares "
        "upstream dependencies with ``deps=[...]`` and returns a ``MaterializeResult`` "
        "with write statistics as Dagster metadata."
    ),
    category="resource_write",
)
def test_cookbook_direct_write() -> None:
    """Show the basic @asset(deps=[...]) + database.write() + MaterializeResult pattern."""
    # --- cookbook:start ---
    import polars as pl
    from dagster import AssetExecutionContext, Definitions, MaterializeResult, asset

    from moncpipelib import PostgresResource

    # Define the resource once -- connection config lives in one place.
    database = PostgresResource(
        host="db.example.com",
        user="writer",
        password="secret",
        database="analytics",
    )

    # Use @asset with deps (not @graph_asset + @op).
    # The asset owns its write target explicitly via database.write().
    @asset(deps=["source_claims"])
    def patient_claims(
        context: AssetExecutionContext,
        database: PostgresResource,
    ) -> MaterializeResult:
        # Transform your data
        df = pl.DataFrame(
            {
                "claim_id": ["C-001", "C-002", "C-003"],
                "patient_id": ["P-001", "P-001", "P-002"],
                "amount": [150.00, 75.50, 200.00],
            }
        )

        # Write directly -- target is always "schema.table"
        result = database.write(
            df,
            target="silver.patient_claims",
            context=context,
        )

        # Return metadata so Dagster UI shows write stats
        return MaterializeResult(metadata=result.to_dagster_metadata())

    defs = Definitions(
        assets=[patient_claims],
        resources={"database": database},
    )

    print("Resource-first write pattern:")
    print("  @asset(deps=['source_claims'])")
    print("  def patient_claims(context, database):")
    print("      result = database.write(df, target='silver.patient_claims', ...)")
    print("      return MaterializeResult(metadata=result.to_dagster_metadata())")
    print()
    print("Advantages over IO manager pattern:")
    print("  - No @graph_asset / @op boilerplate")
    print("  - Explicit target table (no suffix stripping)")
    print("  - Single resource for reads AND writes")
    print("  - Compatible with k8s_job_executor without env var issues")
    # --- cookbook:end ---

    assert defs.get_assets_def("patient_claims") is not None


# ---------------------------------------------------------------------------
# Contract-driven resource write
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Contract-Driven Resource Write",
    description=(
        "When ``contract_search_paths`` is configured on the resource, "
        "``database.write()`` auto-discovers the colocated contract and uses it "
        "for schema validation and write mode configuration. The contract's sink "
        "drives write mode, primary key, SCD2 config, and more -- no explicit "
        "parameters needed on the ``write()`` call."
    ),
    category="resource_write",
)
def test_cookbook_contract_driven_write(tmp_path: Path) -> None:
    """Show database.write() with contract auto-discovery."""
    # Set up a contract file for auto-discovery
    contract_dir = tmp_path / "defs" / "silver" / "patient_claims"
    contract_dir.mkdir(parents=True)
    (contract_dir / "patient_claims.contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "550e8400-e29b-41d4-a716-446655440000"
asset: patient_claims
layer: silver

sinks:
  - type: table
    schema: silver
    table: patient_claims
    mode: upsert                    # write mode from contract
    primary_key: [claim_id]         # upsert conflict key from contract

schema:
  columns:
    - name: claim_id
      type: string
      nullable: false
      primary_key: true
    - name: patient_id
      type: string
      nullable: false
    - name: amount
      type: decimal
      nullable: false
"""
    )

    # --- cookbook:start ---
    from pathlib import Path

    import polars as pl
    from dagster import AssetExecutionContext, Definitions, MaterializeResult, asset

    from moncpipelib import PostgresResource

    # Point the resource at the directory containing contracts.
    # write() auto-discovers *.contract.yaml files by asset name.
    database = PostgresResource(
        host="db.example.com",
        user="writer",
        password="secret",
        database="analytics",
        contract_search_paths=[str(Path(tmp_path) / "defs")],
    )

    @asset(deps=["source_claims"])
    def patient_claims(
        context: AssetExecutionContext,
        database: PostgresResource,
    ) -> MaterializeResult:
        df = pl.DataFrame(
            {
                "claim_id": ["C-001", "C-002"],
                "patient_id": ["P-001", "P-002"],
                "amount": [150.00, 75.50],
            }
        )

        # No write_mode or primary_key needed -- the contract provides them.
        # The contract also validates the DataFrame schema before writing.
        result = database.write(
            df,
            target="silver.patient_claims",
            context=context,
            # write_mode, primary_key auto-discovered from contract
        )
        return MaterializeResult(metadata=result.to_dagster_metadata())

    defs = Definitions(
        assets=[patient_claims],
        resources={"database": database},
    )

    print("Contract-driven write with auto-discovery:")
    print("  database = PostgresResource(")
    print("      ...,")
    print("      contract_search_paths=['defs/'],")
    print("  )")
    print()
    print("  result = database.write(df, target='silver.patient_claims', context=context)")
    print()
    print("The contract provides:")
    print("  - write_mode: upsert (from sink mode)")
    print("  - primary_key: [claim_id] (from sink primary_key)")
    print("  - Schema validation (column types, nullability)")
    print("  - Pipeline ID for lineage tracking")
    # --- cookbook:end ---

    assert defs.get_assets_def("patient_claims") is not None


# ---------------------------------------------------------------------------
# Upsert with explicit primary key
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Upsert with Explicit Primary Key",
    description=(
        "Perform an upsert write using explicit ``write_mode`` and ``primary_key`` "
        "parameters. This is useful when no contract exists or when you want to "
        "override the contract's write mode. Existing rows matching the primary "
        "key are updated; new rows are inserted."
    ),
    category="resource_write",
)
def test_cookbook_upsert_explicit() -> None:
    """Show database.write() with explicit write_mode and primary_key."""
    # --- cookbook:start ---
    import polars as pl
    from dagster import AssetExecutionContext, Definitions, MaterializeResult, asset

    from moncpipelib import PostgresResource

    database = PostgresResource(
        host="db.example.com",
        user="writer",
        password="secret",
        database="analytics",
    )

    @asset(deps=["provider_staging"])
    def dim_provider(
        context: AssetExecutionContext,
        database: PostgresResource,
    ) -> MaterializeResult:
        df = pl.DataFrame(
            {
                "provider_id": ["NPI-001", "NPI-002", "NPI-003"],
                "name": ["Dr. Smith", "Dr. Jones", "Dr. Patel"],
                "specialty": ["oncology", "radiology", "oncology"],
            }
        )

        # Explicit upsert: INSERT ... ON CONFLICT (provider_id) DO UPDATE
        result = database.write(
            df,
            target="gold.dim_provider",
            context=context,
            write_mode="upsert",
            primary_key=["provider_id"],
        )
        return MaterializeResult(metadata=result.to_dagster_metadata())

    defs = Definitions(
        assets=[dim_provider],
        resources={"database": database},
    )

    print("Explicit upsert write:")
    print("  result = database.write(")
    print("      df,")
    print("      target='gold.dim_provider',")
    print("      context=context,")
    print("      write_mode='upsert',")
    print("      primary_key=['provider_id'],")
    print("  )")
    print()
    print("Supported write_mode values:")
    print("  - 'full_refresh'  (default) -- DELETE/TRUNCATE then INSERT")
    print("  - 'upsert'        -- INSERT ... ON CONFLICT ... DO UPDATE")
    print("  - 'append'        -- INSERT only (no dedup)")
    print("  - 'scd2'          -- Slowly Changing Dimension Type 2")
    print("  - 'partition_scoped' -- DELETE partition slice then INSERT")
    # --- cookbook:end ---

    assert defs.get_assets_def("dim_provider") is not None


# ---------------------------------------------------------------------------
# Inspecting WriteResult
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Inspecting WriteResult",
    description=(
        "The ``write()`` method returns a ``WriteResult`` with statistics, contract "
        "status, and lineage info. You can inspect it programmatically for logging, "
        "alerts, or conditional logic. Use ``to_dagster_metadata()`` to convert it "
        "to a dict of ``MetadataValue`` entries for the Dagster UI."
    ),
    category="resource_write",
)
def test_cookbook_write_result() -> None:
    """Show accessing WriteResult fields."""
    # --- cookbook:start ---
    from moncpipelib import WriteResult
    from moncpipelib.io_managers import WriteMode

    # WriteResult is returned by database.write().
    # Here we construct one manually to demonstrate its fields.
    result = WriteResult(
        table_name="gold.dim_provider",
        schema="gold",
        layer="gold",
        write_mode=WriteMode.UPSERT,
        stats={"rows_upserted": 1500},
        row_count=1500,
        batch_count=1,
        lineage_id="01912345-6789-7abc-def0-123456789abc",
        lineage_key="gold.dim_provider/run-abc123",
        columns=["provider_id", "name", "specialty"],
        primary_key=["provider_id"],
    )

    # Access individual fields
    print(f"Table:      {result.table_name}")
    print(f"Schema:     {result.schema}")
    print(f"Layer:      {result.layer}")
    print(f"Write mode: {result.write_mode.value}")
    print(f"Rows:       {result.row_count}")
    print(f"Batches:    {result.batch_count}")
    print(f"Columns:    {result.columns}")
    print(f"Primary key:{result.primary_key}")
    print()

    # Mode-specific stats vary by write_mode:
    #   full_refresh: rows_deleted, rows_inserted, clear_method, insert_method
    #   upsert:       rows_upserted
    #   append:       rows_inserted, insert_method
    #   scd2:         rows_new, rows_expired, rows_inserted, rows_unchanged
    print(f"Stats:      {result.stats}")
    print()

    # Lineage tracking (when enable_row_lineage=True on the resource)
    print(f"Lineage ID: {result.lineage_id}")
    print(f"Lineage key:{result.lineage_key}")
    print()

    # Contract validation summary (when a contract was validated)
    print(f"Contract:   {result.contract_summary}")
    print()

    # Convert to Dagster metadata for MaterializeResult
    metadata = result.to_dagster_metadata()
    print("Dagster metadata keys:")
    for key in sorted(metadata.keys()):
        val = metadata[key]
        if hasattr(val, "value"):
            print(f"  {key}: {val.value}")
        else:
            print(f"  {key}: <{type(val).__name__}>")
    # --- cookbook:end ---

    assert result.table_name == "gold.dim_provider"
    assert result.write_mode == WriteMode.UPSERT
    assert result.row_count == 1500
    assert "write_mode" in metadata
    assert "target_table" in metadata
    assert "row_count" in metadata


# ---------------------------------------------------------------------------
# Shared connection config (Resource + IO Manager)
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Shared Connection Config",
    description=(
        "Define ``PostgresResource`` once and share connection credentials between "
        "direct writes and IO managers. New assets use ``database.write()``; "
        "legacy assets continue using the IO manager. Both share the same host, "
        "port, user, password, and database -- no duplication."
    ),
    category="resource_write",
)
def test_cookbook_shared_config() -> None:
    """Show Definitions pattern where both database and IO manager share config."""
    # --- cookbook:start ---
    import polars as pl
    from dagster import AssetExecutionContext, Definitions, MaterializeResult, asset

    from moncpipelib import PostgresResource
    from moncpipelib.io_managers import PostgresIOManager

    # Shared connection config -- define credentials once.
    # Resource for new assets using database.write() AND shared with IO manager.
    database = PostgresResource(
        host="db.example.com",
        user="writer",
        password="secret",  # noqa: S105  # use EnvVar("DB_PASSWORD") in production
        database="analytics",
        contract_search_paths=["defs/"],
    )

    # IO manager for legacy assets (shares the same PostgresResource)
    pg_io = PostgresIOManager(
        postgres_resource=database,
        default_schema="silver",
    )

    # NEW asset: uses database.write() directly
    @asset(deps=["provider_staging"])
    def dim_provider(
        context: AssetExecutionContext,
        database: PostgresResource,
    ) -> MaterializeResult:
        df = pl.DataFrame({"provider_id": ["NPI-001"], "name": ["Dr. Smith"]})
        result = database.write(df, target="gold.dim_provider", context=context)
        return MaterializeResult(metadata=result.to_dagster_metadata())

    # LEGACY asset: continues to use IO manager (no changes needed)
    @asset(
        io_manager_key="pg_io",
        metadata={"target_schema": "silver"},
    )
    def patient_claims(_context: AssetExecutionContext) -> pl.DataFrame:
        return pl.DataFrame({"claim_id": ["C-001"], "amount": [100.0]})

    # Wire both resource and IO manager in a single Definitions
    defs = Definitions(
        assets=[dim_provider, patient_claims],
        resources={
            "database": database,
            "pg_io": pg_io,
        },
    )

    print("Shared connection config:")
    print(f"  host:     {database.host}")
    print(f"  database: {database.database}")
    print()
    print("Resources:")
    print("  database  -> PostgresResource  (new assets, database.write())")
    print("  pg_io     -> PostgresIOManager (legacy assets, return DataFrame)")
    print()
    print("Migration strategy:")
    print("  1. Define PostgresResource once with all connection config")
    print("  2. Pass postgres_resource=database to PostgresIOManager")
    print("  3. New assets use database.write() + MaterializeResult")
    print("  4. Legacy assets continue using IO manager unchanged")
    print("  5. Migrate legacy assets incrementally (optional)")
    # --- cookbook:end ---

    assert defs.get_assets_def("dim_provider") is not None
    assert defs.get_assets_def("patient_claims") is not None


# ---------------------------------------------------------------------------
# Generate contract checks from PostgresResource
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Generate Contract Checks from Resource",
    description=(
        "Generate Dagster asset checks directly from a ``PostgresResource`` instance "
        "using ``make_contract_checks()``. This recursively scans a directory for "
        "``*.contract.yaml`` files and generates schema validation checks. Connection "
        "credentials are resolved at check execution time (``EnvVar`` compatible)."
    ),
    category="resource_write",
)
def test_cookbook_make_contract_checks(tmp_path: Path) -> None:
    """Show database.make_contract_checks()."""
    # Set up contract files for discovery
    claims_dir = tmp_path / "defs" / "silver" / "claims"
    claims_dir.mkdir(parents=True)
    (claims_dir / "patient_claims.contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "550e8400-e29b-41d4-a716-446655440000"
asset: patient_claims
layer: silver

sinks:
  - type: table
    schema: silver
    table: patient_claims
    mode: upsert

schema:
  columns:
    - name: claim_id
      type: string
      nullable: false
      primary_key: true
      pii: false
    - name: amount
      type: decimal
      nullable: false
      pii: false
"""
    )

    providers_dir = tmp_path / "defs" / "gold" / "providers"
    providers_dir.mkdir(parents=True)
    (providers_dir / "dim_provider.contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "660e8400-e29b-41d4-a716-446655440001"
asset: dim_provider
layer: gold

sinks:
  - type: table
    schema: gold
    table: dim_provider
    mode: full_refresh

schema:
  columns:
    - name: provider_id
      type: string
      nullable: false
      pii: false
    - name: name
      type: string
      nullable: false
      pii: false
"""
    )

    # --- cookbook:start ---
    from pathlib import Path

    from moncpipelib import PostgresResource

    database = PostgresResource(
        host="db.example.com",
        user="writer",
        password="secret",
        database="analytics",
    )

    # Recursively scan defs/ for *.contract.yaml files and generate
    # Dagster asset checks for each contract found.
    defs_path = Path(tmp_path) / "defs"
    checks = database.make_contract_checks(defs_path)

    print(f"Contract checks discovered: {len(checks)}")
    for chk in checks:
        for key in chk.check_keys:
            print(f"  - {key.name} (asset: {key.asset_key.to_user_string()})")

    # Wire into Definitions alongside assets:
    # defs = Definitions(
    #     assets=[...],
    #     asset_checks=checks,
    #     resources={"database": database},
    # )

    print()
    print("make_contract_checks() features:")
    print("  - Recursively scans directory tree")
    print("  - Connection resolved at check execution time (EnvVar safe)")
    print("  - Auto-sets contract_search_paths on the resource")
    print("  - Each contract generates schema validation checks")
    # --- cookbook:end ---

    assert len(checks) == 2
