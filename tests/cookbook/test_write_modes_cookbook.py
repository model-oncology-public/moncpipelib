"""Cookbook tests for upsert and SCD2 write mode configuration.

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
# Upsert examples
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Configure Upsert Mode via Asset Metadata",
    description=(
        "Use Dagster asset metadata to configure upsert (INSERT ON CONFLICT UPDATE) "
        "writes. The IO manager matches incoming rows against the ``primary_key`` "
        "columns -- existing rows are updated while new rows are inserted. "
        "Optionally restrict which columns are updated with ``update_columns``."
    ),
    category="write_modes",
)
def test_upsert_via_metadata() -> None:
    """Demonstrate upsert write mode configured through asset metadata."""
    # --- cookbook:start ---
    import polars as pl
    from dagster import AssetExecutionContext, Definitions, asset

    from moncpipelib.io_managers import PostgresIOManager
    from moncpipelib.resources import PostgresResource

    # Configure the resource with connection config
    database = PostgresResource(
        host="db.example.com",
        user="writer",
        password="secret",
        database="analytics",
    )

    # Configure the IO manager (defaults to full_refresh write mode)
    io_manager = PostgresIOManager(
        postgres_resource=database,
        default_schema="silver",
    )

    # Set write_mode and primary_key in asset metadata to enable upsert.
    # The IO manager reads these at write time.
    @asset(
        io_manager_key="silver_io_manager",
        metadata={
            "write_mode": "upsert",
            "primary_key": ["patient_id"],
            # Optional: only update these columns on conflict (default: all non-PK)
            # "update_columns": ["name", "status"],
        },
    )
    def patients_silver(_context: AssetExecutionContext) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "patient_id": ["PAT-001", "PAT-002", "PAT-003"],
                "name": ["Alice", "Bob", "Charlie"],
                "status": ["active", "active", "discharged"],
            }
        )

    defs = Definitions(
        assets=[patients_silver],
        resources={"silver_io_manager": io_manager},
    )

    # At write time the IO manager executes:
    #   INSERT INTO silver.patients_silver (patient_id, name, status)
    #   VALUES %s
    #   ON CONFLICT (patient_id) DO UPDATE SET name=EXCLUDED.name, status=EXCLUDED.status

    print("Upsert asset metadata keys:")
    print('  write_mode: "upsert"')
    print('  primary_key: ["patient_id"]')
    print()
    print("SQL pattern: INSERT ... ON CONFLICT (patient_id) DO UPDATE SET ...")
    # --- cookbook:end ---

    # Verify asset was created with correct metadata
    asset_node = defs.get_assets_def("patients_silver")
    specs = list(asset_node.specs)
    assert specs[0].metadata["write_mode"] == "upsert"
    assert specs[0].metadata["primary_key"] == ["patient_id"]


@pytest.mark.cookbook(
    title="Configure Upsert Mode via Data Contract",
    description=(
        "Define the upsert primary key in a YAML data contract instead of asset "
        "metadata. Mark columns with ``primary_key: true`` in the schema and set "
        "``mode: upsert`` on the sink entry. The IO manager reconciles the contract "
        "against the resolved write config at write time -- no metadata needed on "
        "the asset definition itself."
    ),
    category="write_modes",
)
def test_upsert_via_contract(tmp_path: Path) -> None:
    """Demonstrate upsert write mode configured through a data contract."""
    # --- cookbook:start ---
    from moncpipelib.contracts import (
        Column,
        ColumnType,
        DataContract,
        Schema,
        load_contract,
    )

    # Option A: Define the contract in Python
    contract = DataContract(
        version="1.0",
        pipeline_id="550e8400-e29b-41d4-a716-446655440000",
        asset="patients_silver",
        layer="silver",
        schema=Schema(
            columns=[
                Column(
                    name="patient_id",
                    type=ColumnType.STRING,
                    nullable=False,
                    primary_key=True,  # drives the upsert conflict key
                ),
                Column(name="name", type=ColumnType.STRING, nullable=False),
                Column(name="status", type=ColumnType.STRING, nullable=False),
            ],
        ),
        sinks=[
            {
                "type": "table",
                "schema": "silver",
                "table": "patients_silver",
                "mode": "upsert",  # write mode declared in the contract
            }
        ],
    )

    pk_columns = contract.get_primary_key_columns()
    print("Contract-defined primary key:", pk_columns)
    print("Contract sink mode:", contract.sinks[0]["mode"])
    # --- cookbook:end ---

    assert pk_columns == ["patient_id"]
    assert contract.sinks[0]["mode"] == "upsert"

    # --- cookbook:start ---

    # Option B: Load the same contract from YAML
    yaml_content = """\
version: "1.0"
pipeline_id: "550e8400-e29b-41d4-a716-446655440000"
asset: patients_silver
layer: silver

schema:
  columns:
    - name: patient_id
      type: string
      nullable: false
      primary_key: true
    - name: name
      type: string
      nullable: false
    - name: status
      type: string
      nullable: false

sinks:
  - type: table
    schema: silver
    table: patients_silver
    mode: upsert
"""
    # --- cookbook:end ---

    # Write the YAML to a temp file so we can load it
    contract_file = tmp_path / "patients_silver.contract.yaml"
    contract_file.write_text(yaml_content)

    # --- cookbook:start ---
    # Save YAML as patients_silver.contract.yaml, then load:
    # load_contract() validates the YAML structure automatically.
    loaded = load_contract(contract_file)

    print("Loaded PK columns:", loaded.get_primary_key_columns())
    print("Loaded sink mode:", loaded.sinks[0]["mode"])

    # With enforce_contracts="error" on the IO manager, the contract's
    # mode and primary_key are reconciled at write time:
    #   - Contract sets mode=upsert -> IO manager uses upsert
    #   - Contract sets primary_key: true on patient_id -> IO manager uses ["patient_id"]
    #   - No metadata needed on the @asset decorator
    # --- cookbook:end ---

    assert loaded.get_primary_key_columns() == ["patient_id"]
    assert loaded.sinks[0]["mode"] == "upsert"


# ---------------------------------------------------------------------------
# SCD2 examples
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Configure SCD2 Mode via Asset Metadata",
    description=(
        "Use Dagster asset metadata to configure SCD2 (Slowly Changing Dimension "
        "Type 2) writes. The IO manager tracks entity versions by comparing a "
        "row hash of ``tracked_columns`` (or all non-business-key columns if "
        "omitted). Changed records are expired and new versions inserted "
        "atomically. Set ``detect_deletes: true`` to expire entities absent "
        "from the incoming DataFrame."
    ),
    category="write_modes",
)
def test_scd2_via_metadata() -> None:
    """Demonstrate SCD2 write mode configured through asset metadata."""
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

    io_manager = PostgresIOManager(
        postgres_resource=database,
        default_schema="silver",
    )

    @asset(
        io_manager_key="silver_io_manager",
        metadata={
            "write_mode": "scd2",
            "business_key": ["product_id"],
            # Optional: only track changes to specific columns
            # Default (omitted): hash ALL non-business-key columns
            "tracked_columns": ["product_name", "price", "category"],
            # Optional: expire records whose business_key is absent from
            # the incoming DataFrame. USE WITH CAUTION -- assumes the
            # incoming data is a complete set of active records.
            "detect_deletes": False,
            # Optional: override SCD2 column names (defaults shown)
            # "effective_from_col": "effective_from",
            # "effective_to_col": "effective_to",
            # "is_current_col": "is_current",
            # "hash_col": "row_hash",
        },
    )
    def dim_product(_context: AssetExecutionContext) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "product_id": ["PROD-001", "PROD-002"],
                "product_name": ["Widget", "Gadget"],
                "price": [9.99, 24.99],
                "category": ["hardware", "electronics"],
            }
        )

    defs = Definitions(
        assets=[dim_product],
        resources={"silver_io_manager": io_manager},
    )

    # The IO manager automatically:
    # 1. Computes row_hash over tracked_columns (or all non-BK columns)
    # 2. Stages incoming data in a temp table
    # 3. Expires changed records (effective_to=now(), is_current=false)
    # 4. Inserts new versions of changed + entirely new records
    # 5. Optionally expires absent records (if detect_deletes=true)

    print("SCD2 asset metadata keys:")
    print('  write_mode: "scd2"')
    print('  business_key: ["product_id"]')
    print('  tracked_columns: ["product_name", "price", "category"]')
    print("  detect_deletes: False")
    print()
    print("Target table columns managed by the IO manager:")
    print("  effective_from  -- when this version became active")
    print("  effective_to    -- when this version was superseded (NULL if current)")
    print("  is_current      -- boolean flag for the active version")
    print("  row_hash        -- SHA-256 hash for change detection")
    # --- cookbook:end ---

    asset_node = defs.get_assets_def("dim_product")
    specs = list(asset_node.specs)
    assert specs[0].metadata["write_mode"] == "scd2"
    assert specs[0].metadata["business_key"] == ["product_id"]
    assert specs[0].metadata["tracked_columns"] == ["product_name", "price", "category"]


@pytest.mark.cookbook(
    title="Configure SCD2 Mode via Data Contract",
    description=(
        "Define SCD2 configuration in a YAML data contract. The sink entry "
        "declares ``mode: scd2`` along with ``business_key``, ``tracked_columns``, "
        "and ``detect_deletes``. The contract loader validates that column "
        "references in ``business_key`` and ``tracked_columns`` exist in the "
        "schema. At write time the IO manager reconciles these against asset "
        "metadata, raising a ``ContractViolationError`` on conflicts."
    ),
    category="write_modes",
)
def test_scd2_via_contract(tmp_path: Path) -> None:
    """Demonstrate SCD2 write mode configured through a data contract."""
    # --- cookbook:start ---
    from moncpipelib.contracts import (
        Column,
        ColumnType,
        DataContract,
        Schema,
        load_contract,
    )

    # Option A: Define the contract in Python
    contract = DataContract(
        version="1.0",
        pipeline_id="550e8400-e29b-41d4-a716-446655440000",
        asset="dim_product",
        layer="silver",
        schema=Schema(
            columns=[
                Column(name="product_id", type=ColumnType.STRING, nullable=False),
                Column(name="product_name", type=ColumnType.STRING, nullable=False),
                Column(name="price", type=ColumnType.DECIMAL, nullable=False),
                Column(name="category", type=ColumnType.STRING, nullable=True),
            ],
        ),
        sinks=[
            {
                "type": "table",
                "schema": "silver",
                "table": "dim_product",
                "mode": "scd2",
                "business_key": ["product_id"],
                "tracked_columns": ["product_name", "price", "category"],
                "detect_deletes": False,
            }
        ],
    )

    sink = contract.sinks[0]
    print("Contract sink configuration:")
    print(f"  mode: {sink['mode']}")
    print(f"  business_key: {sink['business_key']}")
    print(f"  tracked_columns: {sink['tracked_columns']}")
    print(f"  detect_deletes: {sink['detect_deletes']}")
    # --- cookbook:end ---

    assert sink["mode"] == "scd2"
    assert sink["business_key"] == ["product_id"]

    # --- cookbook:start ---

    # Option B: Load the same contract from YAML
    yaml_content = """\
version: "1.0"
pipeline_id: "550e8400-e29b-41d4-a716-446655440000"
asset: dim_product
layer: silver

schema:
  columns:
    - name: product_id
      type: string
      nullable: false
    - name: product_name
      type: string
      nullable: false
    - name: price
      type: decimal
      nullable: false
    - name: category
      type: string
      nullable: true

sinks:
  - type: table
    schema: silver
    table: dim_product
    mode: scd2
    business_key: [product_id]
    tracked_columns: [product_name, price, category]
    detect_deletes: false
"""
    # --- cookbook:end ---

    contract_file = tmp_path / "dim_product.contract.yaml"
    contract_file.write_text(yaml_content)

    # --- cookbook:start ---
    # Save YAML as dim_product.contract.yaml, then load:
    # load_contract() validates the YAML structure automatically.
    loaded = load_contract(contract_file)

    loaded_sink = loaded.sinks[0]
    print(f"\nLoaded mode: {loaded_sink['mode']}")
    print(f"Loaded business_key: {loaded_sink['business_key']}")
    print(f"Loaded tracked_columns: {loaded_sink['tracked_columns']}")
    print(f"Loaded detect_deletes: {loaded_sink['detect_deletes']}")

    # With enforce_contracts="error" on the IO manager:
    #   - Contract drives mode, business_key, tracked_columns, detect_deletes
    #   - The @asset only needs: metadata={"write_mode": "scd2"}
    #   - tracked_columns must also be set in metadata (explicit acknowledgement)
    #   - business_key is silently overridden from the contract if not in metadata
    # --- cookbook:end ---

    assert loaded_sink["mode"] == "scd2"
    assert loaded_sink["business_key"] == ["product_id"]
    assert loaded_sink["tracked_columns"] == ["product_name", "price", "category"]
    assert loaded_sink["detect_deletes"] is False


@pytest.mark.cookbook(
    title="SCD2 Row Hash Computation for Change Detection",
    description=(
        "The IO manager automatically computes a SHA-256 row hash over "
        "``tracked_columns`` (or all non-business-key columns) before writing. "
        "This hash is compared against existing records to determine which rows "
        "have changed. You can also compute the hash yourself using "
        "``compute_row_hash`` for debugging or pre-validation."
    ),
    category="write_modes",
)
def test_scd2_row_hash() -> None:
    """Demonstrate SCD2 row hash computation."""
    # --- cookbook:start ---
    import polars as pl

    from moncpipelib.transforms.hashing import compute_row_hash

    df = pl.DataFrame(
        {
            "product_id": ["PROD-001", "PROD-002", "PROD-003"],
            "product_name": ["Widget", "Gadget", "Widget"],
            "price": [9.99, 24.99, 9.99],
            "category": ["hardware", "electronics", "hardware"],
        }
    )

    # Hash only the tracked columns (not the business key)
    tracked = ["product_name", "price", "category"]
    df_with_hash = df.with_columns(compute_row_hash(tracked))

    print("DataFrame with row_hash:")
    for row in df_with_hash.iter_rows(named=True):
        print(f"  {row['product_id']}: {row['row_hash'][:16]}...")

    # Identical tracked values produce identical hashes
    hash_1 = df_with_hash.row(0, named=True)["row_hash"]
    hash_3 = df_with_hash.row(2, named=True)["row_hash"]
    print(f"\nPROD-001 hash == PROD-003 hash: {hash_1 == hash_3}")
    print("(Same tracked values -> same hash -> no SCD2 version change)")
    # --- cookbook:end ---

    assert hash_1 == hash_3
    hash_2 = df_with_hash.row(1, named=True)["row_hash"]
    assert hash_1 != hash_2


@pytest.mark.cookbook(
    title="SCD2 Historical Backfill with effective_date",
    description=(
        "Use the ``effective_date`` parameter on ``database.write()`` to load "
        "historical SCD2 data with correct period boundaries. Without it, all "
        "inserts and expirations are stamped with ``now()``. With it, you can "
        "backfill from periodic source files (quarterly, semi-annual) and "
        "produce an accurate version history."
    ),
    category="write_modes",
)
def test_scd2_effective_date() -> None:
    """Demonstrate SCD2 historical backfill with effective_date."""
    # --- cookbook:start ---
    # When loading historical SCD2 data from periodic source files,
    # pass effective_date to stamp rows with the correct period boundary
    # instead of the current timestamp.
    #
    # Example: loading three semi-annual snapshots:
    #
    #   database.write(
    #       h1_2025_df,
    #       target="silver.products",
    #       context=ctx,
    #       effective_date=date(2025, 1, 1),
    #   )
    #   database.write(
    #       h2_2025_df,
    #       target="silver.products",
    #       context=ctx,
    #       effective_date=date(2025, 7, 1),
    #   )
    #   database.write(
    #       current_df,
    #       target="silver.products",
    #       context=ctx,
    #       # omit effective_date for current data -> uses now()
    #   )
    #
    # Result: a clean SCD2 history with correct period boundaries:
    #
    #   product_id | name   | effective_from | effective_to | is_current
    #   PROD-001   | Widget | 2025-01-01     | 2025-07-01   | false
    #   PROD-001   | Widget | 2025-07-01     | NULL         | true
    # The effective_date parameter is accepted by write()
    import inspect
    from datetime import date

    from moncpipelib.resources.postgres import PostgresResource

    sig = inspect.signature(PostgresResource.write)
    param = sig.parameters["effective_date"]
    print(f"Parameter: {param.name}")
    print("Type: date | None")
    print(f"Default: {param.default}")

    # It replaces now() in scd2_finalize for both:
    # - effective_from on newly inserted rows
    # - effective_to on expired rows
    print("\nLoads must be in chronological order for coherent history.")
    print("Omit effective_date (or pass None) for current-period loads.")

    # effective_date is a date object
    d = date(2025, 1, 1)
    print(f"\nExample: effective_date={d}")
    # --- cookbook:end ---

    assert param.default is None
    assert "effective_date" in sig.parameters


@pytest.mark.cookbook(
    title="SCD2 Period Manifest and Partition Mapping",
    description=(
        "Define historical periods in a dedicated ``*.source.yaml`` file and use "
        "``build_partitions_from_periods()`` to generate Dagster partitions. "
        "Each partition maps to a period with a source and effective date range. "
        "Use ``get_period_for_partition()`` inside the asset to resolve the "
        "current partition back to a Period."
    ),
    category="write_modes",
)
def test_scd2_period_manifest(tmp_path: Path) -> None:
    """Demonstrate period manifest with Dagster partition mapping."""
    (tmp_path / "products.source.yaml").write_text(
        """\
source_id: "e5f6a7b8-c9d0-1234-ef01-23456789abcd"
source_name: products-source
description: Product data source
periods:
  - source: "https://example.com/data-2025-h1.csv"
    effective_from: 2025-01-01
    effective_to: 2025-07-01
  - source: "https://example.com/data-2025-h2.csv"
    effective_from: 2025-07-01
    effective_to: 2026-01-01
  - source: "/data/current.csv"
    effective_from: 2026-01-01
    effective_to:
"""
    )
    (tmp_path / "contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: products
layer: silver
data_source: products.source.yaml
schema:
  strict: false
  columns:
    - name: product_id
      type: string
      nullable: false
      pii: false
"""
    )

    # --- cookbook:start ---
    from moncpipelib.contracts import load_contract, load_data_source
    from moncpipelib.historical import (
        build_partitions_from_periods,
        get_period_for_partition,
    )

    # Load the data source directly (or via contract.data_source)
    source = load_data_source(tmp_path / "products.source.yaml")

    # Periods are parsed from the *.source.yaml file
    print(f"Periods defined: {len(source.periods)}")
    for p in source.periods:
        print(f"  {p.effective_from} -> {p.effective_to or 'current'}: {p.source}")

    # Generate Dagster partitions from periods
    partitions_def = build_partitions_from_periods(source)
    keys = partitions_def.get_partition_keys()
    print(f"\nPartition keys: {keys}")

    # Resolve a partition key back to a Period
    period = get_period_for_partition(source, "2025-07-01")
    print(f"\nPartition '2025-07-01' -> source: {period.source}")

    # Contract also resolves data_source automatically
    contract = load_contract(tmp_path / "contract.yaml")
    assert contract.data_source is not None
    print(f"\nContract data_source: {contract.data_source.source_id}")

    # In a Dagster asset:
    #
    #   source = load_data_source("products.source.yaml")
    #   @asset(partitions_def=build_partitions_from_periods(source))
    #   def my_scd2_asset(context, database: PostgresResource):
    #       period = get_period_for_partition(source, context.partition_key)
    #       df = pl.read_csv(period.source)
    #       return database.write(
    #           df, target="silver.products", context=context,
    #           effective_date=period.effective_from,
    #       )
    # --- cookbook:end ---

    assert len(source.periods) == 3
    assert len(keys) == 3
    assert period.source == "https://example.com/data-2025-h2.csv"


@pytest.mark.cookbook(
    title="SCD2 Period Partition Keys for Scoped Backfills",
    description=(
        "Add ``partition_key`` to each period in the ``*.source.yaml`` file to "
        "enable partition-scoped SCD2 writes during historical backfills. "
        "moncpipelib automatically injects the partition key as a DataFrame "
        "column using the sink's ``partition_column`` name. This avoids using "
        "the managed ``effective_from`` column (which doesn't exist in the "
        "user's DataFrame) as the partition column."
    ),
    category="write_modes",
)
def test_scd2_period_partition_key(tmp_path: Path) -> None:
    """Demonstrate partition_key on periods for scoped SCD2 backfills."""
    (tmp_path / "products.source.yaml").write_text(
        """\
source_id: "f6a7b8c9-d0e1-2345-f012-3456789abcde"
source_name: products-partitioned
description: Product data source with partition keys
periods:
  - source: "https://example.com/h1-2025.csv"
    effective_from: 2025-01-01
    effective_to: 2025-07-01
    partition_key: "2025-H1"
  - source: "https://example.com/h2-2025.csv"
    effective_from: 2025-07-01
    effective_to: 2026-01-01
    partition_key: "2025-H2"
  - source: "/data/current.csv"
    effective_from: 2026-01-01
    partition_key: "2026-current"
"""
    )
    (tmp_path / "contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: products
layer: silver
data_source: products.source.yaml
schema:
  strict: false
  columns:
    - name: product_id
      type: string
      nullable: false
      pii: false
sinks:
  - type: table
    schema: silver
    table: products
    mode: scd2
    business_key:
      - product_id
    partition_column: load_period
"""
    )

    # --- cookbook:start ---
    import polars as pl

    from moncpipelib.contracts import load_contract, load_data_source
    from moncpipelib.historical import (
        build_partitions_from_periods,
        get_period_for_partition,
    )

    # Load the data source (periods live here now)
    source = load_data_source(tmp_path / "products.source.yaml")

    # partition_key values become Dagster partition keys
    partitions_def = build_partitions_from_periods(source)
    keys = partitions_def.get_partition_keys()
    print(f"Partition keys: {keys}")

    # Resolve partition key back to a Period
    period = get_period_for_partition(source, "2025-H2")
    print(f"Period '2025-H2': source={period.source}, from={period.effective_from}")

    # During write(), moncpipelib injects the partition_key as a column:
    #
    #   source = load_data_source("products.source.yaml")
    #   @asset(partitions_def=build_partitions_from_periods(source))
    #   def my_asset(context, database: PostgresResource):
    #       period = get_period_for_partition(source, context.partition_key)
    #       df = pl.read_csv(period.source)
    #       # No need to add 'load_period' column -- moncpipelib injects it
    #       return database.write(
    #           df, target="silver.products", context=context,
    #           effective_date=period.effective_from,
    #       )
    #
    # The injected column passes partition validation and scopes the
    # SCD2 merge to only compare against rows from this period.

    # Verify injection works via the resource method directly
    from moncpipelib.resources.postgres import PostgresResource

    contract = load_contract(tmp_path / "contract.yaml")
    resource = PostgresResource(
        host="localhost", port=5432, database="test", user="test", password="test"
    )
    df = pl.DataFrame({"product_id": ["P001"], "name": ["Widget"]})
    injected = resource._inject_period_partition_column(
        df,
        {"partition_column": "load_period"},
        contract,
        period.effective_from,
    )
    print(f"\nInjected columns: {injected.columns}")
    print(f"load_period value: {injected['load_period'][0]}")
    # --- cookbook:end ---

    assert keys == ["2025-H1", "2025-H2", "2026-current"]
    assert period.source == "https://example.com/h2-2025.csv"
    assert "load_period" in injected.columns
    assert injected["load_period"][0] == "2025-H2"


@pytest.mark.cookbook(
    title="Contract Validation Catches Invalid Sink Configuration",
    description=(
        "The contract loader validates sink field types and cross-references "
        "column names against the schema. Invalid mode values, non-string "
        "business keys, and references to undefined columns are caught at "
        "load time -- before any data is written."
    ),
    category="write_modes",
)
def test_contract_sink_validation() -> None:
    """Demonstrate contract validation catching sink misconfigurations."""
    # --- cookbook:start ---
    from moncpipelib.contracts import validate_contract_schema

    # Example 1: Unknown mode value
    bad_mode = {
        "version": "1.0",
        "pipeline_id": "550e8400-e29b-41d4-a716-446655440000",
        "asset": "test",
        "layer": "silver",
        "sinks": [
            {
                "type": "table",
                "schema": "silver",
                "table": "test",
                "mode": "upert",  # typo!
            }
        ],
    }
    errors = validate_contract_schema(bad_mode)
    print("Bad mode errors:")
    for e in errors:
        if "mode" in e:
            print(f"  {e}")

    # Example 2: business_key references a column not in the schema
    bad_ref = {
        "version": "1.0",
        "pipeline_id": "550e8400-e29b-41d4-a716-446655440000",
        "asset": "test",
        "layer": "silver",
        "schema": {
            "columns": [
                {"name": "id", "type": "string"},
                {"name": "name", "type": "string"},
            ],
        },
        "sinks": [
            {
                "type": "table",
                "schema": "silver",
                "table": "test",
                "mode": "scd2",
                "business_key": ["id", "missing_col"],
            }
        ],
    }
    errors = validate_contract_schema(bad_ref)
    print("\nBad column reference errors:")
    for e in errors:
        if "references" in e:
            print(f"  {e}")
    # --- cookbook:end ---

    mode_errors = [e for e in validate_contract_schema(bad_mode) if "mode" in e]
    assert len(mode_errors) == 1
    assert "upert" in mode_errors[0]

    ref_errors = [e for e in validate_contract_schema(bad_ref) if "references" in e]
    assert len(ref_errors) == 1
    assert "missing_col" in ref_errors[0]


# ---------------------------------------------------------------------------
# Streaming batch examples
# ---------------------------------------------------------------------------


@pytest.mark.cookbook(
    title="Stream Large Datasets with BatchedDataFrame",
    description=(
        "Use ``BatchedDataFrame`` and ``transform_batched`` to process datasets "
        "that exceed available memory. The IO manager consumes the iterator "
        "batch-by-batch, writing each batch within a single transaction. Each "
        "batch is discarded after writing, keeping memory usage constant "
        "regardless of total dataset size."
    ),
    category="streaming",
)
def test_batched_dataframe_basics() -> None:
    """Demonstrate creating and inspecting a BatchedDataFrame."""
    # --- cookbook:start ---
    import polars as pl

    from moncpipelib.streaming import BatchedDataFrame, transform_batched

    # Simulate a source that yields data in batches (e.g., database cursor,
    # file reader, API pagination). In production this would come from
    # PostgresResource.read_batched() or similar.
    def generate_batches():
        for batch_num in range(3):
            start = batch_num * 1000
            yield pl.DataFrame(
                {
                    "id": list(range(start, start + 1000)),
                    "value": [f"row_{i}" for i in range(start, start + 1000)],
                }
            )

    # Option 1: Wrap an iterator directly
    batched = BatchedDataFrame(
        batches=generate_batches(),
        total_rows_hint=3000,  # optional, used for progress logging
    )
    print(f"BatchedDataFrame created (hint: {batched.total_rows_hint} rows)")

    # Option 2: Apply a per-batch transform using transform_batched
    def clean_batch(df: pl.DataFrame) -> pl.DataFrame:
        """Transform applied independently to each batch."""
        return df.with_columns(
            pl.col("value").str.to_uppercase().alias("value"),
            pl.lit("cleaned").alias("status"),
        )

    batched_with_transform = transform_batched(
        generate_batches(),
        transform_fn=clean_batch,
        total_rows_hint=3000,
    )

    # Consume the first batch to verify the transform
    first_batch = next(iter(batched_with_transform.batches))
    print(f"\nFirst batch shape: {first_batch.shape}")
    print(f"Columns: {first_batch.columns}")
    print(f"Sample row: {first_batch.row(0, named=True)}")
    # --- cookbook:end ---

    assert first_batch.shape == (1000, 3)
    assert "status" in first_batch.columns
    assert first_batch.row(0, named=True)["value"] == "ROW_0"
    assert first_batch.row(0, named=True)["status"] == "cleaned"


@pytest.mark.cookbook(
    title="Batched Upsert Pipeline Pattern",
    description=(
        "Combine ``PostgresResource.read_batched()`` with ``transform_batched`` "
        "and a ``BatchedDataFrame`` return type to build an end-to-end streaming "
        "upsert pipeline. The IO manager writes each batch using "
        "INSERT ... ON CONFLICT UPDATE, so the full dataset never needs to be in "
        "memory. Compatible with ``full_refresh``, ``append``, ``upsert``, "
        "and ``scd2`` write modes."
    ),
    category="streaming",
)
def test_batched_upsert_pipeline_pattern() -> None:
    """Demonstrate a streaming upsert pipeline using BatchedDataFrame."""
    # --- cookbook:start ---
    import polars as pl
    from dagster import AssetExecutionContext, Definitions, asset

    from moncpipelib.io_managers import PostgresIOManager
    from moncpipelib.resources import PostgresResource
    from moncpipelib.streaming import BatchedDataFrame, transform_batched

    # Resources
    source_db = PostgresResource(
        host="source-db.example.com",
        user="reader",
        password="secret",
        database="source_system",
    )
    target_db = PostgresResource(
        host="target-db.example.com",
        user="writer",
        password="secret",
        database="analytics",
    )
    io_manager = PostgresIOManager(
        postgres_resource=target_db,
        default_schema="silver",
    )

    # Per-batch transform (applied independently to each batch)
    def transform_claims(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col("claim_id").str.strip_chars().alias("claim_id"),
            pl.col("amount").cast(pl.Decimal(precision=10, scale=2)),
        )

    # The asset reads in batches, transforms each, and returns a
    # BatchedDataFrame. The IO manager writes each batch as an upsert.
    @asset(
        io_manager_key="silver_io_manager",
        required_resource_keys={"source_db"},
        metadata={
            "write_mode": "upsert",
            "primary_key": ["claim_id"],
        },
    )
    def claims_silver(_context: AssetExecutionContext) -> BatchedDataFrame:
        # In production, _context.resources.source_db.read_batched(...)
        # returns an Iterator[pl.DataFrame] from a server-side cursor.
        return transform_batched(
            iter([]),  # placeholder for source_db.read_batched(...)
            transform_fn=transform_claims,
            total_rows_hint=500_000,
        )

    defs = Definitions(
        assets=[claims_silver],
        resources={
            "source_db": source_db,
            "silver_io_manager": io_manager,
        },
    )

    # At runtime the IO manager:
    # 1. Opens a single transaction
    # 2. Validates the first batch against the contract (if configured)
    # 3. Writes each batch: INSERT ... ON CONFLICT (claim_id) DO UPDATE SET ...
    # 4. Commits after all batches are written
    # 5. Records rows_written and batches_written on the BatchedDataFrame

    print("Batched upsert pipeline:")
    print("  Source: PostgresResource.read_batched() -> Iterator[pl.DataFrame]")
    print("  Transform: transform_batched(batches, transform_fn)")
    print("  Sink: PostgresIOManager (write_mode=upsert, primary_key=[claim_id])")
    print()
    print("Compatible write modes for BatchedDataFrame:")
    print("  - full_refresh (DELETE/TRUNCATE then INSERT each batch)")
    print("  - append (INSERT each batch)")
    print("  - upsert (INSERT ON CONFLICT each batch)")
    print("  - scd2 (stage then finalize change detection)")
    # --- cookbook:end ---

    asset_node = defs.get_assets_def("claims_silver")
    specs = list(asset_node.specs)
    assert specs[0].metadata["write_mode"] == "upsert"
    assert specs[0].metadata["primary_key"] == ["claim_id"]


@pytest.mark.cookbook(
    title="Streaming SCD2 with BatchedDataFrame",
    description=(
        "SCD2 write mode works with ``BatchedDataFrame`` for large dimension "
        "tables that cannot fit in memory. Each batch is staged into a temp "
        "table with its ``row_hash`` computed, and change detection (expire "
        "changed, insert new versions) runs once after all batches are consumed. "
        "Use ``detect_deletes=True`` only when the stream represents the "
        "complete set of active records."
    ),
    category="streaming",
)
def test_scd2_batched_pipeline_pattern() -> None:
    """Demonstrate SCD2 with BatchedDataFrame for large dimension tables."""
    # --- cookbook:start ---
    import polars as pl
    from dagster import AssetExecutionContext, Definitions, asset

    from moncpipelib.streaming import BatchedDataFrame, transform_batched

    # Asset that streams a large dimension table in batches with SCD2 tracking.
    # Each batch is staged into a temp table; change detection runs after all
    # batches are consumed.
    @asset(
        metadata={
            "write_mode": "scd2",
            "business_key": ["product_id"],
            # Optional: only track specific columns for change detection.
            # If omitted, all non-business-key columns are hashed.
            "tracked_columns": ["name", "price", "category"],
            # detect_deletes=True expires records whose business key is
            # absent from the incoming data. Only use when the stream
            # represents the COMPLETE set of active records.
            # "detect_deletes": True,
        },
    )
    def dim_product(_context: AssetExecutionContext) -> BatchedDataFrame:
        # In production: db.read_batched("SELECT ...", batch_size=50_000)
        batches = iter(
            [
                pl.DataFrame(
                    {
                        "product_id": ["P001", "P002"],
                        "name": ["Widget", "Gadget"],
                        "price": [9.99, 19.99],
                        "category": ["tools", "electronics"],
                    }
                ),
                pl.DataFrame(
                    {
                        "product_id": ["P003"],
                        "name": ["Doohickey"],
                        "price": [4.99],
                        "category": ["misc"],
                    }
                ),
            ]
        )
        return transform_batched(batches, transform_fn=lambda df: df)

    defs = Definitions(assets=[dim_product])
    asset_node = defs.get_assets_def("dim_product")
    specs = list(asset_node.specs)

    print("SCD2 batched pipeline pattern:")
    print(f"  Asset: {specs[0].key.to_user_string()}")
    print(f"  write_mode: {specs[0].metadata['write_mode']}")
    print(f"  business_key: {specs[0].metadata['business_key']}")
    print(f"  tracked_columns: {specs[0].metadata['tracked_columns']}")
    print()

    # How it works:
    # 1. First batch: create staging table, compute row_hash, insert into staging
    # 2. Each subsequent batch: compute row_hash, insert into staging
    # 3. After all batches: finalize change detection
    #    - Count new and changed records vs current dimension state
    #    - Expire changed records (set is_current=false, effective_to=now())
    #    - Insert new versions of changed + entirely new records
    #    - Optionally expire absent business keys (detect_deletes)
    print("Execution flow:")
    print("  Batch 1 -> create staging table, hash + stage 2 rows")
    print("  Batch 2 -> hash + stage 1 row")
    print("  Finalize -> count, expire changed, insert new versions")
    # --- cookbook:end ---

    assert specs[0].metadata["write_mode"] == "scd2"
    assert specs[0].metadata["business_key"] == ["product_id"]
    assert specs[0].metadata["tracked_columns"] == ["name", "price", "category"]


@pytest.mark.cookbook(
    title="Period Registry: Cross-Location Partition Discovery",
    description=(
        "The period registry enables bronze and silver pipelines in separate "
        "Dagster code locations to share partition definitions via a shared "
        "database table (``lineage.period_registry``). When a write includes "
        "a contract with ``data_source`` and an ``effective_date``, the period "
        "is automatically registered as ``materialized``. You can also call "
        "``register_period()`` explicitly for standalone registration."
    ),
    category="write-modes",
)
def test_period_registry_pattern() -> None:
    """Demonstrate the period registry for cross-location partition discovery."""
    # --- cookbook:start ---
    import inspect

    from moncpipelib.resources.postgres import PostgresResource

    # register_period() is a public method on PostgresResource for explicit
    # period registration. It performs an INSERT ... ON CONFLICT upsert so
    # re-registration is safe.
    sig = inspect.signature(PostgresResource.register_period)
    params = list(sig.parameters.keys())

    print("register_period() signature:")
    print(f"  Parameters: {params}")
    print()

    # The method accepts these arguments:
    #   source_id      - UUID from DataSource.source_id
    #   partition_key  - e.g. "2025Q1"
    #   effective_from - start date of the period
    #   effective_to   - optional end date
    #   source_uri     - optional URL/URI for the source data
    #   status         - default "materialized"
    #   source_name    - human-readable name from DataSource.source_name
    #   registered_by  - optional identifier (e.g., asset name)
    #   run_id         - optional Dagster run ID for audit tracking
    #   metadata       - optional JSON dict for extra context

    # Example usage (requires a live database connection):
    #   database.register_period(
    #       source_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    #       partition_key="2025Q1",
    #       effective_from=date(2025, 1, 1),
    #       effective_to=date(2025, 3, 31),
    #       source_uri="https://cms.gov/files/asp-crosswalk-2025q1.zip",
    #       status="materialized",
    #       source_name="cms_asp_crosswalk",
    #       registered_by="bronze__cms_asp_crosswalk",
    #       run_id="01234567-89ab-cdef-0123-456789abcdef",
    #   )

    # Auto-registration happens automatically during write() when:
    # 1. The contract has a data_source (loaded from *.source.yaml)
    # 2. effective_date is passed to write()
    # 3. effective_date matches a period in data_source.periods
    #
    # The period is registered as "materialized" after a successful commit.
    # Failures are logged as warnings and do not block the write.

    print("Auto-registration triggers:")
    print("  1. Contract has data_source with periods")
    print("  2. effective_date passed to write()")
    print("  3. effective_date matches a period's effective_from")
    print()
    print("Status lifecycle:")
    print("  registered   -> initial state (inserted by source pipeline)")
    print("  materialized -> data has been written to the target table")
    # --- cookbook:end ---

    assert "source_id" in params
    assert "source_name" in params
    assert "partition_key" in params
    assert "effective_from" in params
    assert "status" in params
    assert "run_id" in params
