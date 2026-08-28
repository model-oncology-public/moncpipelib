# SCD Type 2 Guide

## Overview

moncpipelib supports Slowly Changing Dimension Type 2 (SCD2) as a write mode
in the PostgresIOManager. SCD2 tracks historical changes to dimension data by
expiring old row versions and inserting new ones, rather than overwriting in
place.

The IO manager handles all SCD2 mechanics automatically:

- Computing a deterministic row hash from tracked columns
- Staging incoming data in a temp table
- Detecting new and changed records via hash comparison
- Expiring changed records (setting `effective_to` and `is_current = false`)
- Inserting new versions with `effective_from = now()` and `is_current = true`

All of this runs within a single database transaction -- there is no window
where a business key has zero or two current rows.

## Quick Start

### 1. Create the target table

```sql
CREATE TABLE silver.dim_product (
    id              bigint GENERATED ALWAYS AS IDENTITY,
    product_id      text NOT NULL,       -- business key
    product_name    text,                -- tracked column
    category        text,                -- tracked column
    price           numeric,             -- tracked column
    row_hash        text NOT NULL,       -- SHA-256 of tracked columns
    effective_from  timestamptz NOT NULL DEFAULT now(),
    effective_to    timestamptz,         -- NULL = current version
    is_current      boolean NOT NULL DEFAULT true,
    _lineage_id     text,
    _lineage_key    text
);

-- Enforces exactly one current row per business key
CREATE UNIQUE INDEX uq_dim_product_current
    ON silver.dim_product (product_id) WHERE (is_current);
```

### 2. Define the asset

```python
@asset(
    metadata={
        "write_mode": "scd2",
        "business_key": "product_id",
        "tracked_columns": ["product_name", "category", "price"],
    },
)
def dim_product(context, bronze_products: pl.DataFrame) -> pl.DataFrame:
    # Return clean data -- IO manager handles the rest
    return bronze_products.select(
        "product_id", "product_name", "category", "price"
    )
```

The IO manager will:

1. Add `_lineage_id` and `_lineage_key` columns (if lineage is enabled)
2. Compute `row_hash` as SHA-256 of `product_name | category | price`
3. Stage the DataFrame into a temp table
4. Expire changed rows and insert new versions (within one transaction)
5. Report stats in Dagster UI metadata

## Configuration

### Required metadata

| Key | Type | Description |
|-----|------|-------------|
| `write_mode` | `"scd2"` | Enables SCD2 mode |
| `business_key` | `str` or `list[str]` | Column(s) uniquely identifying a business entity |

### Optional metadata

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `tracked_columns` | `list[str]` | All non-key, non-lineage columns | Columns to include in hash |
| `effective_from_col` | `str` | `"effective_from"` | Column name for version start |
| `effective_to_col` | `str` | `"effective_to"` | Column name for version end |
| `is_current_col` | `str` | `"is_current"` | Column name for current flag |
| `hash_col` | `str` | `"row_hash"` | Column name for row hash |
| `detect_deletes` | `bool` | `False` | Expire records absent from incoming data (see warning below) |

> **WARNING -- DATA LOSS RISK:** `detect_deletes` assumes the incoming DataFrame
> contains the **complete** set of currently active records. Every business key
> NOT present in the incoming data will be expired. **DO NOT** enable this for
> partial or filtered loads (e.g., only today's changed records). Doing so will
> incorrectly expire every record not in the current batch. An empty DataFrame
> with `detect_deletes=True` will raise a `ValueError` to prevent accidental
> expiration of the entire dimension table.

### Composite business keys

```python
@asset(metadata={
    "write_mode": "scd2",
    "business_key": ["region", "product_id"],
})
```

## Standalone Utilities

### `compute_row_hash`

A Polars expression for deterministic SHA-256 hashing, usable outside the IO
manager:

```python
from moncpipelib import compute_row_hash

df = df.with_columns(
    compute_row_hash(["col_a", "col_b", "col_c"]).alias("row_hash")
)
```

Properties:

- Deterministic across sessions and platforms (SHA-256, not Polars `.hash()`)
- Null-safe: nulls are replaced with a sentinel before hashing
- Column order matters: `["a", "b"]` produces a different hash than `["b", "a"]`
- Type-safe: all values are cast to string before hashing

### `detect_changes`

Python-side change detection for logging or custom merge logic:

```python
from moncpipelib.scd import detect_changes

result = detect_changes(
    incoming=new_df,
    current=current_df,
    business_key="product_id",
)
print(result.summary)
# {"new": 5, "changed": 3, "unchanged": 92, "deleted": 0, "total_incoming": 100}
```

This is informational only -- the IO manager's Postgres-side write handles the
actual expiration and insertion atomically within a single transaction.

## Target Table Requirements

SCD2 target tables must have these columns (names are configurable):

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `row_hash` | `text` | NO | SHA-256 hash of tracked columns |
| `effective_from` | `timestamptz` | NO | When this version became effective |
| `effective_to` | `timestamptz` | YES | When this version was superseded (NULL = current) |
| `is_current` | `boolean` | NO | Convenience flag for current version |

Recommended indexes:

```sql
-- Partial unique index: one current row per business key
CREATE UNIQUE INDEX uq_{table}_current
    ON {schema}.{table} ({business_key}) WHERE (is_current);

-- For historical lookups
CREATE INDEX idx_{table}_effective
    ON {schema}.{table} ({business_key}, effective_from, effective_to);
```

The `WHERE (is_current)` predicate is load-bearing, not a refinement. A plain
(non-partial) unique index that instead includes `is_current` as a key column,
such as `UNIQUE (business_key, is_current)`, tolerates the first expiry of any
business key but raises `UniqueViolation` the second time that key changes:
there can only be one `(key, false)` row. The table works in initial testing
and fails on a later, unrelated run. Since #401 the SCD2 writer runs a
catalog preflight at write time and logs a warning (once per target per
process) when it finds this shape.

## Dagster UI Metadata

After each SCD2 write, the following metadata appears in the Dagster UI:

| Key | Description |
|-----|-------------|
| `write_mode` | `"scd2"` |
| `business_key` | Business key column(s) |
| `rows_new` | Records with entirely new business keys |
| `rows_expired` | Previously-current records that were expired |
| `rows_inserted` | Total records inserted (new + changed) |
| `rows_unchanged` | Records where hash matched (no action taken) |
| `rows_deleted` | Records expired by `detect_deletes` (0 when disabled) |

## Limitations

- **No BatchedDataFrame support.** SCD2 requires the full DataFrame for
  change detection. Dimension tables should typically fit in memory.
- **Delete detection is opt-in.** By default, the IO manager does **not**
  expire records whose business key is absent from incoming data. Set
  `detect_deletes: True` in metadata to enable this for full-snapshot loads.
  See the warning in the configuration section above.
- **Floating-point precision.** Float columns may cause false-positive changes
  due to precision differences. Round or normalize floats before they reach
  the SCD2 asset, or exclude them from `tracked_columns`.
- **Concurrent writes.** Two simultaneous SCD2 writes to the same table may
  produce unexpected results. PostgreSQL row-level locks prevent duplicate
  expirations, but the behavior is not guaranteed to be correct.

## How It Works

The IO manager executes two statements within a single transaction:

```sql
-- 1. Expire changed rows
UPDATE target t
SET effective_to = now(), is_current = false
FROM _scd2_staging s
WHERE t.business_key = s.business_key
  AND t.is_current = true
  AND t.row_hash <> s.row_hash;

-- 2. Insert new and changed rows
--    After the UPDATE, changed rows no longer have is_current=true,
--    so the LEFT JOIN treats them the same as genuinely new records.
INSERT INTO target (..., effective_from, is_current)
SELECT ..., now(), true
FROM _scd2_staging s
LEFT JOIN target t
    ON t.business_key = s.business_key AND t.is_current = true
WHERE t.business_key IS NULL;

-- 3. (Only when detect_deletes=True) Expire absent business keys
UPDATE target t
SET effective_to = now(), is_current = false
WHERE t.is_current = true
  AND NOT EXISTS (
      SELECT 1 FROM _scd2_staging s
      WHERE t.business_key = s.business_key
  );
```

The UPDATE expires old versions of changed records. The INSERT then adds new
versions for both changed and genuinely new records. Both statements use
`now()` (which returns the transaction timestamp), ensuring `effective_to` of
the expired row exactly matches `effective_from` of the new row. Because all
statements run within the same transaction, no intermediate states are visible
to other sessions, and the partial unique index is respected.

When `detect_deletes` is enabled, a third UPDATE expires any current records
whose business key does not appear in the staging table. This is safe only
when the incoming data is a complete snapshot of all active records.
