# Best Practices for moncpipelib ETL Pipelines

This guide covers recommended patterns for building efficient, maintainable data pipelines with moncpipelib.

## 1. Data Transforms - Memory Optimization

All transform functions accept any input type (String, Int, Float, Boolean, Null dtype, etc.) and handle conversion internally. **Do NOT pre-cast columns before using transforms** - this wastes memory by creating intermediate copies.

```python
# BAD - wastes memory by creating intermediate String column
df.with_columns([
    pl.col("price").cast(pl.String).pipe(safe_decimal).alias("price_clean")
])

# GOOD - transforms accept any input type directly
df.with_columns([
    safe_decimal("price"),
])
```

### Available Transforms

| Function | Output Type | Description |
|----------|-------------|-------------|
| `clean_text(col)` | String | Strips whitespace, converts empty to null |
| `safe_decimal(col)` | Float64 | Parses any type, null for invalid |
| `safe_int(col)` | Int64 | Parses any type, null for invalid |
| `safe_bool(col)` | Boolean | Handles t/f/true/false/1/0/yes/no |
| `safe_date(col, format)` | Date | Parses with format string |
| `safe_datetime(col, format)` | Datetime | Parses with format string |

## 2. Custom Python Functions with map_elements()

When using custom functions with `map_elements()`, handle type conversion inside the function rather than pre-casting:

```python
import re
import polars as pl

# GOOD - function handles conversion internally
def normalize_ndc(ndc) -> str | None:
    if ndc is None:
        return None
    digits = re.sub(r"[^0-9]", "", str(ndc))  # str() handles any input
    # ... rest of normalization logic
    return normalized

# Then use without pre-cast:
df.with_columns([
    pl.col("ndc_code").map_elements(normalize_ndc, return_dtype=pl.String)
])
```

## 3. Lazy Evaluation for Large DataFrames

Use `.lazy()` / `.collect()` for memory-efficient transformations on large datasets:

```python
# GOOD - processes data lazily, optimizes query plan
df_transformed = (
    input_df.lazy()
    .select([
        clean_text("name"),
        safe_decimal("price"),
        safe_bool("is_active"),
    ])
    .collect()
)
```

Benefits:
- Polars optimizes the query plan before execution
- Reduces memory usage by not materializing intermediate results
- Can push down filters and projections

## 4. IO Manager Configuration

`PostgresIOManager` delegates all database operations to a `PostgresResource`. Define the resource once and pass it in:

```python
from moncpipelib import PostgresResource, PostgresIOManager

database = PostgresResource(
    host=EnvVar("DB_HOST"),
    port=EnvVar.int("DB_PORT"),
    user=EnvVar("DB_USER"),
    password=EnvVar("DB_PASSWORD"),
    database=EnvVar("DB_NAME"),
)

PostgresIOManager(
    postgres_resource=database,
    default_schema="synthetic_silver",
)
```

### Parameter Reference

| Parameter | Required | Description |
|-----------|----------|-------------|
| `postgres_resource` | Yes | `PostgresResource` instance for all DB operations |
| `default_schema` | No | Fallback schema when not specified per-asset |
| `write_mode` | No | Default write mode (full_refresh, upsert, append, scd2) |

Performance tuning (batch sizes, bulk insert methods, full refresh strategy) is configured on `PostgresResource` directly. See [cookbook.md](cookbook.md) for working examples.

## 5. Output Column Normalization

Apply `clean_text()` to text columns in final output for consistent null/whitespace handling:

```python
from moncpipelib import clean_text, safe_decimal

return df.select([
    "id_column",
    clean_text("name"),
    clean_text("description"),
    safe_decimal("amount"),
    "is_active",  # Boolean columns don't need clean_text
])
```

This ensures:
- Consistent whitespace handling across all text fields
- Empty strings converted to null for proper SQL semantics
- No leading/trailing whitespace in output data

## 6. Performance Tuning (Large Tables)

For tables with >10k rows, configure performance options on the `PostgresResource`:

```python
from moncpipelib import PostgresResource

PostgresResource(
    host=EnvVar("DB_HOST"), port=EnvVar.int("DB_PORT"),
    user=EnvVar("DB_USER"), password=EnvVar("DB_PASSWORD"),
    database=EnvVar("DB_NAME"),
    full_refresh_method="truncate",  # O(1) vs DELETE's O(n)
    bulk_insert_method="copy",       # 4-5x faster than execute_values
    insert_chunk_size=100_000,       # Process in chunks to limit memory
)
```

### Performance Options

| Option | Default | When to Change |
|--------|---------|----------------|
| `full_refresh_method` | `auto` | Use `truncate` for large tables where you can tolerate ACCESS EXCLUSIVE lock |
| `bulk_insert_method` | `auto` | Use `copy` for large append/full_refresh operations |
| `insert_chunk_size` | `None` (auto) | Set explicitly for very large DataFrames to control memory |

### Trade-offs

**full_refresh_method:**
- `delete`: ROW EXCLUSIVE lock, allows concurrent reads, O(n) performance
- `truncate`: ACCESS EXCLUSIVE lock, blocks all access, O(1) performance

**bulk_insert_method:**
- `execute_values`: Required for upsert mode, more flexible
- `copy`: 4-5x faster, only works for append/full_refresh modes

### Planner Statistics for Partitioned Targets (analyze_after_write)

PostgreSQL's autovacuum autoanalyzes ordinary tables and leaf partitions, but
never a partitioned parent -- it has no storage of its own, so the per-table
autoanalyze thresholds can never fire on it. A partitioned target that is only
ever written through moncpipelib will show `pg_class.reltuples = -1` on the
parent forever, and any query planned against the parent (cross-partition
scans, `count(*)`, `max(col)`) runs with missing statistics.

The write path closes this gap with a post-commit `ANALYZE` step, controlled
by `analyze_after_write` (resource field, per-asset metadata, or `write()`
parameter):

| Value | Behavior |
|-------|----------|
| `partitioned` (default) | `ANALYZE` only when the target is a partitioned parent |
| `always` | `ANALYZE` the target after every changed write |
| `never` | No post-write `ANALYZE` |

Details:

- Runs after the write transaction commits, in its own short transaction, so
  it never extends the write's locks.
- Skipped when the write changed no rows.
- SCD2 targets are skipped: the SCD2 writer already refreshes target
  statistics in-transaction as part of its change-detection planning.
- On PostgreSQL 18+ a partitioned parent uses `ANALYZE ONLY`, which refreshes
  the parent's aggregate statistics without rewriting per-leaf statistics
  (leaves stay owned by autovacuum). Older servers fall back to a plain
  recursive `ANALYZE` -- on wide partition trees consider `never` plus a
  scheduled maintenance `ANALYZE` if that cost shows up in write latency.
- `ANALYZE` requires table ownership or the `MAINTAIN` privilege (PG17+). A
  failed `ANALYZE` logs a warning and never fails the committed write.
- The action taken is surfaced in the write stats / Dagster output metadata
  as `analyze_after_write` (`parent`, `recursive`, or `table`).

`VACUUM` and `FREEZE` deliberately stay out of the write path; they belong to
autovacuum and scheduled maintenance.

## 7. Memory-Efficient Database Reads

For large database tables, `pl.read_database()` can use 20-30x the final DataFrame size in peak memory. Use `PostgresResource`'s batched read methods to stream results via server-side cursors instead.

### When to Use Batched Reads

| Table Size | Approach |
|------------|----------|
| < 100k rows | `pl.read_database()` is fine |
| 100k - 1M rows | Consider `read_batched_to_dataframe()` |
| > 1M rows | Use `read_batched_to_dataframe()` or process batches individually |

### Basic Usage

```python
@asset
def my_asset(context, database: PostgresResource) -> pl.DataFrame:
    # Reads in 50k-row batches via server-side cursor, concatenates at the end
    return database.read_batched_to_dataframe(
        "SELECT * FROM large_table",
        context=context,
    )
```

### Processing Batches Individually

For maximum memory efficiency, process each batch before accumulating:

```python
@asset
def my_asset(context, database: PostgresResource) -> pl.DataFrame:
    chunks = []
    for batch in database.read_batched(
        "SELECT * FROM large_table",
        batch_size=50_000,
        context=context,
    ):
        # Filter or transform each batch to reduce memory
        chunks.append(batch.filter(pl.col("active") == True))
    return pl.concat(chunks)
```

### Tuning batch_size

- **Default (50,000)** works well for most tables
- **Reduce** for wide tables (many columns or large text fields) to keep each batch small in memory
- **Increase** for narrow tables (few small columns) to reduce round-trip overhead

### Standalone Functions

The `read_batched` and `read_batched_to_dataframe` functions are also available as top-level imports for use outside of `PostgresResource`:

```python
from moncpipelib import read_batched, read_batched_to_dataframe

# Works with PostgresResource, SQLAlchemy Engine/Connection, or psycopg connection
df = read_batched_to_dataframe("SELECT * FROM large_table", database, context=context)
```

When passing a raw `psycopg` connection, use `method="offset"` with an `order_by` column instead of the default streaming method.

## 8. Streaming Writes for Large Datasets

For datasets with millions of rows, traditional single-DataFrame writes require memory proportional to the full dataset size. `BatchedDataFrame` enables batch-by-batch writing with **constant memory usage** regardless of dataset size.

### Memory Impact Comparison

| Row Count | DataFrame Size | Traditional Peak Memory | Streaming Peak Memory | Improvement |
|-----------|---------------|------------------------|----------------------|-------------|
| 438k | 164 MB | 848 MB (4.4x) | ~300 MB | 2.8x |
| 1M | 373 MB | 1.8 GB | ~300 MB | 6x |
| 5M | 1.9 GB | 8.5 GB | ~500 MB | 17x |
| 10M | 3.7 GB | 16.5 GB | ~500 MB | 33x |
| 20M | 7.5 GB | 33 GB | ~500 MB | 66x |

### When to Use Streaming Writes

| Dataset Size | Approach | Why |
|-------------|----------|-----|
| < 1M rows | Standard `pl.DataFrame` | Memory overhead is acceptable |
| 1M - 5M rows | Consider `BatchedDataFrame` | Significant memory savings |
| > 5M rows | Use `BatchedDataFrame` | Essential for large datasets |

### The Only Pattern for BatchedDataFrame: @asset

`BatchedDataFrame` wraps a Python generator. Generators **cannot be pickled**, which means they cannot be serialized between ops. With k8s executor, Dagster serializes intermediate op outputs to shared storage -- even within a `@graph_asset`. This makes multi-op patterns incompatible with `BatchedDataFrame`.

**Always use `@asset` for streaming pipelines.** A single `@asset` runs as one step in any executor, `io_manager_key` works natively, and there is no intermediate serialization.

> **WARNING**: Do NOT use `@graph_asset` with `BatchedDataFrame`. Any pattern that passes a `BatchedDataFrame` between ops (including single-op `@graph_asset`) will fail with either a pickle error or an `io_manager_key` propagation issue. This has been validated through multiple failed attempts (PRs #176, #177, #185 in data-platform).

```python
from moncpipelib import (
    BatchedDataFrame, PostgresResource, transform_batched,
    safe_int, safe_decimal, safe_date, clean_text,
)

@asset(io_manager_key="silver_io_manager")
def large_table_silver(
    context: AssetExecutionContext,
    database: PostgresResource,
) -> BatchedDataFrame:
    """Extract and transform large table with constant memory usage."""
    batches = database.read_batched(
        "SELECT * FROM bronze.large_table",
        batch_size=50_000,
        context=context,
    )
    return transform_batched(batches, transform_fn=transform_batch)


def transform_batch(batch: pl.DataFrame) -> pl.DataFrame:
    """Transform a single batch - called once per batch."""
    return (
        batch.lazy()
        .select([
            safe_int("account_id"),
            clean_text("customer_name"),
            safe_decimal("amount"),
            safe_date("transaction_date"),
        ])
        .collect()
    )
```

For multi-op pipelines that return `pl.DataFrame` (gold layer with joins, lookups, etc.), use `@op/@graph_asset` instead. See [Section 12](#12-kubernetes-execution-opgraph_asset-pattern) for that pattern.

### Key Points

**Benefits:**
- Peak memory becomes O(batch_size) instead of O(total_rows)
- Process arbitrary dataset sizes within fixed memory limits
- All write modes supported (full_refresh, upsert, append, scd2)
- Single transaction ensures atomicity
- Progress logging shows percentage complete

**Constraints:**
- Transform function must work on individual batches independently
- Cross-batch operations (global deduplication, percentiles) not supported
- Use `read_batched_to_dataframe()` for transforms requiring full dataset

### Traditional vs Streaming Comparison

**Traditional (full materialization):**
```python
@asset
def large_table_extract(context, database: PostgresResource) -> pl.DataFrame:
    # Materializes entire dataset in memory
    return database.read_batched_to_dataframe(
        "SELECT * FROM bronze.large_table",
        context=context,
    )

@asset(io_manager_key="silver_io_manager")
def large_table_silver(large_table_extract: pl.DataFrame) -> pl.DataFrame:
    # Holds input + output DataFrames simultaneously
    return large_table_extract.select([...])
```

**Streaming (constant memory):**
```python
@asset(io_manager_key="silver_io_manager")
def large_table_silver(
    context: AssetExecutionContext,
    database: PostgresResource,
) -> BatchedDataFrame:
    """Single asset processes data in batches without materialization."""
    batches = database.read_batched(
        "SELECT * FROM bronze.large_table",
        batch_size=50_000,
        context=context,
    )
    return transform_batched(batches, transform_fn=transform_large_batch)

def transform_large_batch(batch: pl.DataFrame) -> pl.DataFrame:
    return batch.select([...])
```

### Tuning batch_size

The `batch_size` parameter controls memory usage per batch:

- **Default (50,000)**: Works well for most tables (~18 MB per batch for typical width)
- **Reduce for wide tables**: Many columns or large text fields → use 25,000 or 10,000
- **Increase for narrow tables**: Few small columns → use 100,000 for better throughput

### Transaction Behavior

All batches are written within a **single transaction**. This ensures atomicity: either all batches commit successfully or none do. If the write fails partway through, the database is automatically rolled back to its pre-write state.

### Lineage and Metadata

Row-level tracking works identically to single-DataFrame writes. The IO manager creates one record for the entire materialization (not per batch) and applies it to all rows across all batches.

- **With lineage enabled** (lineage tracker configured): each row receives `_lineage_id` and `_lineage_key` columns only. Per-layer metadata columns are not added.
- **Without lineage** (default): each row receives per-layer metadata columns (`_{layer}_run_id`, `_{layer}_processed_at`, and optionally `_source_file`).

## 9. Polars Type Reference

Use modern Polars types (avoid deprecated aliases):

```python
# GOOD - modern types
pl.String
pl.Int64
pl.Float64

# BAD - deprecated aliases
pl.Utf8  # Use pl.String instead
```

## 10. Common Patterns

### Bronze Layer - Raw Data Loading

```python
@asset(io_manager_key="bronze_io_manager")
def orders_bronze(context, database: PostgresResource) -> pl.DataFrame:
    context.add_output_metadata({"source_file": "blob://container/orders.csv"})

    with database.get_connection() as conn:
        return pl.read_database("SELECT * FROM raw.orders", connection=conn)
```

For large source tables, use batched reads to avoid out-of-memory errors:

```python
@asset(io_manager_key="bronze_io_manager")
def large_orders_bronze(context, database: PostgresResource) -> pl.DataFrame:
    context.add_output_metadata({"source_file": "source_db/raw.orders"})

    return database.read_batched_to_dataframe(
        "SELECT * FROM raw.orders",
        context=context,
    )
```

### Silver Layer - Cleaning and Normalization

```python
@asset(io_manager_key="silver_io_manager")
def orders_silver(orders_bronze: pl.DataFrame) -> pl.DataFrame:
    return orders_bronze.select([
        clean_text("order_id"),
        clean_text("customer_id"),
        safe_decimal("amount"),
        safe_date("order_date"),
        safe_bool("is_fulfilled"),
    ])
```

### Gold Layer - Business Logic and Aggregations

```python
@asset(io_manager_key="gold_io_manager")
def customer_summary_gold(orders_silver: pl.DataFrame) -> pl.DataFrame:
    return orders_silver.group_by("customer_id").agg([
        pl.sum("amount").alias("total_amount"),
        pl.count("order_id").alias("order_count"),
        pl.max("order_date").alias("last_order_date"),
    ])
```

## 11. IO Manager vs In-Memory Assets

Not every asset needs to persist to the database. Use the `PostgresIOManager` only for assets that should be written to a table.

### When to Use PostgresIOManager

Use `io_manager_key` when the asset:
- Should be persisted to a database table
- Returns a `pl.DataFrame`
- Is a final output or needs to be queryable

```python
# GOOD - persists to synthetic_gold.dim_service table
@asset(io_manager_key="gold_io_manager")
def dim_service_gold(bronze_data: dict) -> pl.DataFrame:
    # Transform and return DataFrame
    return final_df
```

### When NOT to Use PostgresIOManager

Omit `io_manager_key` for intermediate assets that:
- Only pass data to downstream assets
- Return non-DataFrame types (dict, list, tuple, etc.)
- Don't need to be persisted

```python
# GOOD - intermediate extraction, passes data to next asset
@asset  # No io_manager_key - uses Dagster's default in-memory IO
def bronze_service_data(context, database: PostgresResource) -> dict:
    """Extract data from multiple sources for downstream processing."""
    with database.get_connection() as conn:
        ndc_df = pl.read_database("SELECT * FROM bronze.ndc_reference", conn)
        fact_df = pl.read_database("SELECT * FROM bronze.fact_table", conn)

    return {
        "ndc_reference": ndc_df,
        "fact_data": fact_df,
    }

# BAD - will fail with TypeError because dict is not a DataFrame
@asset(io_manager_key="gold_io_manager")  # Don't do this!
def bronze_service_data(context, database: PostgresResource) -> dict:
    ...
```

### Common Error

If you see this error:
```
TypeError: PostgresIOManager.handle_output expected a Polars DataFrame,
but received dict from asset 'bronze_service_data'.
```

The fix is to remove `io_manager_key` from the intermediate asset:
```python
# Before (broken)
@asset(io_manager_key="gold_io_manager")
def bronze_service_data(...) -> dict:

# After (works)
@asset
def bronze_service_data(...) -> dict:
```

### Pattern Summary

| Asset Type | Returns | io_manager_key | Persisted |
|------------|---------|----------------|-----------|
| Intermediate extraction | dict, tuple, list | None (omit) | No (in-memory) |
| Final output | pl.DataFrame | Required | Yes (database) |
| Bronze raw data | pl.DataFrame | Optional | Depends on use case |

## 12. Kubernetes Execution: @op/@graph_asset Pattern

When running Dagster in Kubernetes with `k8s_job_executor`, each `@asset` runs in a separate pod. This creates challenges:

- **mem_io_manager fails**: Stores data in memory, but separate pods don't share memory
- **Serialization overhead**: Data must be persisted between assets
- **Multiple small assets**: Each becomes an independent pod, adding overhead

### Solution: @op and @graph_asset

Use `@op` for ETL steps that should share memory, composed into a single `@graph_asset`:

```python
import polars as pl
from dagster import (
    MetadataValue,
    OpExecutionContext,
    Out,
    graph_asset,
    op,
)
from moncpipelib import PostgresIOManager, PostgresResource

# Ops run in the SAME process, sharing memory directly.
# Intermediate ops do NOT need io_manager_key -- within a graph,
# data passes in memory between ops regardless.
@op(required_resource_keys={"database"})
def extract_source_a(context: OpExecutionContext) -> pl.DataFrame:
    """Extract from source A."""
    with context.resources.database.get_connection() as conn:
        return pl.read_database("SELECT * FROM source_a", conn)

@op(required_resource_keys={"database"})
def extract_source_b(context: OpExecutionContext) -> pl.DataFrame:
    """Extract from source B."""
    with context.resources.database.get_connection() as conn:
        return pl.read_database("SELECT * FROM source_b", conn)

@op
def transform_data(source_a: pl.DataFrame, source_b: pl.DataFrame) -> pl.DataFrame:
    """Transform and merge data."""
    return source_a.join(source_b, on="id")

@op(out=Out(io_manager_key="gold_io_manager"))  # Final op MUST declare io_manager_key
def finalize_output(context: OpExecutionContext, df: pl.DataFrame) -> pl.DataFrame:
    """Add metadata and return for persistence via gold_io_manager."""
    context.add_output_metadata({"num_records": MetadataValue.int(len(df))})
    return df

# Graph asset composes ops into a single materializable unit
@graph_asset
def my_dimension_table() -> pl.DataFrame:
    """Produces dim table - final op's io_manager_key determines persistence."""
    a = extract_source_a()
    b = extract_source_b()
    transformed = transform_data(a, b)
    return finalize_output(transformed)

database = PostgresResource(...)

defs = Definitions(
    assets=[my_dimension_table],
    resources={
        "database": database,
        "gold_io_manager": PostgresIOManager(postgres_resource=database, default_schema="gold"),
    },
)
```

### Pattern Comparison

| Aspect | Multiple @asset | @op/@graph_asset |
|--------|-----------------|------------------|
| Process isolation | Each asset = separate pod | All ops = single pod |
| Intermediate data | Must persist to DB/storage | Shared in memory |
| Serialization | Required between assets | None between ops |
| Logical structure | Maintained | Maintained |
| Memory usage | Lower per-pod | Higher single-pod |
| k8s executor | Compatible (with persistence) | Compatible |

### When to Use Each Pattern

**Use @op/@graph_asset when:**
- ETL steps are tightly coupled and always run together
- Intermediate data doesn't need independent persistence
- You want to avoid serialization overhead
- Steps produce data types that IO managers can't handle (dicts, custom objects)
- All intermediate outputs are **picklable** (pl.DataFrame, dict, list -- NOT BatchedDataFrame)

**Use multiple @asset when:**
- Assets need independent scheduling/materialization
- Intermediate data should be persisted for debugging/reuse
- Assets have different freshness policies
- You need fine-grained observability per step

### Important Notes

- **IO manager key on final op (CRITICAL)**: The terminal op in the graph MUST declare `out=Out(io_manager_key="layer_io_manager")`. `@graph_asset` does NOT accept `io_manager_key` directly -- it uses the terminal op's `Out` to determine which IO manager persists the output. Omitting this causes `DagsterInvalidDefinitionError` because Dagster falls back to the default `"io_manager"` key.
- **Intermediate ops omit io_manager_key**: Extract and transform ops do not need `Out(io_manager_key=...)`.
- **No BatchedDataFrame in @graph_asset**: `BatchedDataFrame` wraps a generator which cannot be pickled. Dagster serializes intermediate op outputs (even within a graph with k8s executor), so passing `BatchedDataFrame` between ops will fail with `TypeError: cannot pickle 'generator' object`. Use `@asset` for streaming pipelines instead (see [Section 8](#8-streaming-writes-for-large-datasets)).
- **Resource access**: Use `required_resource_keys={"database"}` on ops that need database access, then access via `context.resources.database`.
- **Metadata**: Add output metadata in the final op via `context.add_output_metadata()`.

This pattern resolves the `DagsterUnmetExecutorRequirementsError` while maintaining clean, modular ETL code.