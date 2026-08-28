# Row-Level Lineage Tracking

This document describes the row-level lineage tracking system in moncpipelib.

## Overview

The lineage system provides automatic row-level tracking of data as it flows through the medallion architecture (bronze → silver → gold). It uses **UUID7** (time-ordered UUIDs) with composite backup keys for maximum resilience and debuggability.

### Key Features

- **UUID7 Primary Keys**: Time-ordered UUIDs with embedded timestamps for chronological sorting
- **Versioned Composite Backup Keys**: Human-readable keys for recovery scenarios (format: `v1:{asset}:{layer}:{date}:{run_id}`)
- **Timestamp Extraction**: Can extract creation time from UUID7 even if lineage table is unavailable
- **Forward Compatible**: Version prefix allows programmatic parsing even if format evolves
- **Source Tracking**: Files and external systems
- **Temporal Information**: Data dates and date ranges
- **Backfill Operations**: Full history with reasons
- **Aggregation Lineage**: Many:1 relationships via parent tracking
- **Transformation Types**: Capture operation types (aggregate, join, filter, etc.)

## Columns Added to DataFrames

When full lineage mode is enabled (default), the IO Manager adds these columns to your DataFrames:

| Column | Type | Description |
|--------|------|-------------|
| `_lineage_id` | UUID | Foreign key to `lineage.data_lineage` table (UUID7 with embedded timestamp) |
| `_lineage_key` | Text | Human-readable composite key: `v1:{asset}:{layer}:{date}:{run_id}` |

All other metadata (source_file, run_id, processed_at, data_date, parent_lineage_ids, etc.) is stored in the `lineage.data_lineage` table and queryable via the `_lineage_id` foreign key.

## Database Schema

The lineage system requires the `lineage.data_lineage` table to exist in your database. This table stores lineage metadata, while data tables contain `_lineage_id` and `_lineage_key` columns.

### Required Alembic Migration

```python
"""Add lineage tracking support

Revision ID: <your_revision_id>
Revises: <previous_revision>
Create Date: <date>

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '<your_revision_id>'
down_revision = '<previous_revision>'
branch_labels = None
depends_on = None


def upgrade():
    # Create lineage schema
    op.execute('CREATE SCHEMA IF NOT EXISTS lineage')

    # Create data_lineage table
    op.create_table(
        'data_lineage',
        sa.Column('lineage_id', postgresql.UUID(), nullable=False, comment='UUID7 (time-ordered, extractable timestamp)'),
        sa.Column('lineage_key', sa.Text(), nullable=False, comment='Composite backup key: {asset}:{layer}:{date}:{run_id_prefix}'),
        sa.Column('run_id', sa.Text(), nullable=False),
        sa.Column('asset_name', sa.Text(), nullable=False),
        sa.Column('layer', sa.Text(), nullable=False),
        sa.Column('source_file', sa.Text(), nullable=True),
        sa.Column('source_system', sa.Text(), nullable=True),
        sa.Column('data_date', sa.Date(), nullable=True),
        sa.Column('data_date_range', postgresql.DATERANGE(), nullable=True),
        sa.Column('processed_at', sa.TIMESTAMP(), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('row_count', sa.Integer(), nullable=True),
        sa.Column('is_backfill', sa.Boolean(), nullable=False, server_default=sa.text('FALSE')),
        sa.Column('backfill_reason', sa.Text(), nullable=True),
        sa.Column('replaces_lineage_id', postgresql.UUID(), nullable=True),
        sa.Column('parent_lineage_ids', postgresql.ARRAY(postgresql.UUID()), nullable=True),
        sa.Column('transformation_type', sa.Text(), nullable=True),
        sa.Column('created_by', sa.Text(), nullable=True, server_default=sa.text('CURRENT_USER')),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('lineage_id'),
        sa.ForeignKeyConstraint(['replaces_lineage_id'], ['lineage.data_lineage.lineage_id'], ),
        schema='lineage'
    )

    # Create indexes for common queries
    op.create_index(
        'ix_lineage_data_lineage_run_id',
        'data_lineage',
        ['run_id'],
        schema='lineage'
    )
    op.create_index(
        'ix_lineage_data_lineage_asset_name',
        'data_lineage',
        ['asset_name'],
        schema='lineage'
    )
    op.create_index(
        'ix_lineage_data_lineage_layer',
        'data_lineage',
        ['layer'],
        schema='lineage'
    )
    op.create_index(
        'ix_lineage_data_lineage_source_file',
        'data_lineage',
        ['source_file'],
        schema='lineage'
    )
    op.create_index(
        'ix_lineage_data_lineage_data_date',
        'data_lineage',
        ['data_date'],
        schema='lineage'
    )
    op.create_index(
        'ix_lineage_data_lineage_is_backfill',
        'data_lineage',
        ['is_backfill'],
        schema='lineage'
    )
    op.create_index(
        'ix_lineage_data_lineage_processed_at',
        'data_lineage',
        ['processed_at'],
        schema='lineage'
    )
    op.create_index(
        'ix_lineage_data_lineage_lineage_key',
        'data_lineage',
        ['lineage_key'],
        schema='lineage'
    )


def downgrade():
    op.drop_index('ix_lineage_data_lineage_lineage_key', table_name='data_lineage', schema='lineage')
    op.drop_index('ix_lineage_data_lineage_processed_at', table_name='data_lineage', schema='lineage')
    op.drop_index('ix_lineage_data_lineage_is_backfill', table_name='data_lineage', schema='lineage')
    op.drop_index('ix_lineage_data_lineage_data_date', table_name='data_lineage', schema='lineage')
    op.drop_index('ix_lineage_data_lineage_source_file', table_name='data_lineage', schema='lineage')
    op.drop_index('ix_lineage_data_lineage_layer', table_name='data_lineage', schema='lineage')
    op.drop_index('ix_lineage_data_lineage_asset_name', table_name='data_lineage', schema='lineage')
    op.drop_index('ix_lineage_data_lineage_run_id', table_name='data_lineage', schema='lineage')
    op.drop_table('data_lineage', schema='lineage')
    op.execute('DROP SCHEMA IF EXISTS lineage CASCADE')
```

### Adding Lineage Columns to Data Tables

Each data table that uses lineage tracking needs `_lineage_id` and `_lineage_key` columns:

```python
def upgrade():
    # Add lineage ID (UUID7 foreign key)
    op.add_column(
        'your_table_name',
        sa.Column('_lineage_id', postgresql.UUID(), nullable=True),
        schema='your_schema'
    )
    op.create_foreign_key(
        'fk_your_table_lineage',
        'your_table_name',
        'data_lineage',
        ['_lineage_id'],
        ['lineage_id'],
        source_schema='your_schema',
        referent_schema='lineage'
    )
    op.create_index(
        'ix_your_table_lineage_id',
        'your_table_name',
        ['_lineage_id'],
        schema='your_schema'
    )

    # Add lineage key (human-readable composite key)
    op.add_column(
        'your_table_name',
        sa.Column('_lineage_key', sa.Text(), nullable=True),
        schema='your_schema'
    )
    op.create_index(
        'ix_your_table_lineage_key',
        'your_table_name',
        ['_lineage_key'],
        schema='your_schema'
    )


def downgrade():
    op.drop_index('ix_your_table_lineage_key', table_name='your_table_name', schema='your_schema')
    op.drop_column('your_table_name', '_lineage_key', schema='your_schema')
    op.drop_index('ix_your_table_lineage_id', table_name='your_table_name', schema='your_schema')
    op.drop_constraint('fk_your_table_lineage', 'your_table_name', schema='your_schema', type_='foreignkey')
    op.drop_column('your_table_name', '_lineage_id', schema='your_schema')
```

## Usage

### Basic Configuration

Lineage tracking is **enabled by default** (opt-out). It is configured on
`PostgresResource` -- the IO manager delegates all writes (including lineage)
to the resource and has no lineage fields of its own. The layer is resolved
automatically from the target schema when it is a recognized layer name
(bronze/silver/gold), falling back to the contract's `layer:` field:

```python
from dagster import Definitions, EnvVar
from moncpipelib import PostgresIOManager, PostgresResource

database = PostgresResource(
    host=EnvVar("DB_HOST"),
    port=5432,
    user=EnvVar("DB_USER"),
    password=EnvVar("DB_PASSWORD"),
    database=EnvVar("DB_NAME"),
    # enable_row_lineage=True  # Default, can omit
)

defs = Definitions(
    resources={
        "database": database,
        "io_manager": PostgresIOManager(
            postgres_resource=database,
            default_schema="bronze",
        ),
    },
)
```

To **disable** lineage tracking:

```python
database = PostgresResource(
    # ... other config ...
    enable_row_lineage=False,  # Opt out of lineage tracking
)
```

### Test-Mode Isolation for Integration Tests (#420)

Integration-test and ephemeral harnesses typically redirect the *sink* table
to an isolated schema, but the write path's lineage side-effects would still
target the shared `lineage` schema -- including a `silver_materialized` stamp
on the real `period_registry` that makes the environment's sensor silently
skip the first real materialization.

Set the environment variable in the test harness to make every lineage and
period-registry write a logged no-op:

```bash
export MONCPIPELIB_SKIP_LINEAGE_WRITES=1
```

This skips, per write: the `data_lineage` record, `contract_validation_runs`
persistence, the `column_metadata` PII sync, `period_registry` stamping, the
`pipeline_registry` upsert, the `scd2_reconciliations` reconcile audit row,
and OpenLineage emission. The data write itself keeps byte-for-byte
production shape (#424, #426): for a lineage-enabled write, the managed
`_lineage_id` / `_lineage_key` columns are attached with real generated
values, so NOT NULL sink constraints hold (consumer models may declare
either or both NOT NULL, and UPSERT staging tables LIKE-clone those
constraints). The id references no `data_lineage` row: ephemeral test sinks
are dropped after the run, and test harnesses clone target tables with FKs
stripped -- a skip-mode write against a REAL table that enforces the
`data_lineage` FK fails loudly on that FK, which blocks test-isolated
writes from landing in production tables. Layer metadata columns
(`_{layer}_run_id`, `_{layer}_processed_at`, `_source_file`) are only added
where production would add them, i.e. for writes without lineage (no
tracker, no layer, or a `lineage.enabled: false` contract) -- never as a
substitute for the lineage columns. Prior to #424, skip mode incorrectly
swapped writes onto the metadata-columns path, changing the write shape and
breaking column validation against production-shaped targets; #426 fixed
the id from NULL to a real generated value after NOT NULL consumer sinks
rejected NULL.

Each skipped write is logged at WARNING level. This is a test-isolation
switch, not an operational toggle: it disables the compliance audit trail
and must never be set in production deployments. It is deliberately an
environment variable (set by the harness process) rather than a resource or
contract field, so production configuration cannot enable it accidentally.

### Example 1: Bronze Ingestion with Source File

```python
from dagster import asset
from datetime import date
import polars as pl

@asset(
    io_manager_key="bronze_io_manager",
    metadata={
        "source_file": "claims_2024_01_15.csv",
        "source_system": "sftp",
        "data_date": date(2024, 1, 15),
    },
)
def claims_bronze() -> pl.DataFrame:
    # Ingest data from source file
    df = pl.read_csv("path/to/claims_2024_01_15.csv")
    return df
```

This will:
1. Create a lineage record tracking the source file and date
2. Add `_lineage_id` to all rows in the dataframe
3. Write the data with lineage to the `bronze.claims` table

### Example 2: Silver Transformation

```python
@asset(
    io_manager_key="silver_io_manager",
    metadata={
        "transformation_type": "filter",
    },
)
def claims_silver(claims_bronze: pl.DataFrame) -> pl.DataFrame:
    # Transform data
    df = claims_bronze.filter(pl.col("amount") > 0)
    return df
```

This will:
1. Detect parent lineage IDs from `claims_bronze._lineage_id`
2. Create a new lineage record with `parent_lineage_ids` array
3. Attach new `_lineage_id` to the output dataframe
4. Track the transformation type

### Example 3: Gold Aggregation

```python
@asset(
    io_manager_key="gold_io_manager",
    metadata={
        "transformation_type": "aggregate",
    },
)
def claims_summary_gold(claims_silver: pl.DataFrame) -> pl.DataFrame:
    # Aggregate data
    df = claims_silver.group_by("provider_id").agg(
        pl.col("amount").sum().alias("total_amount"),
        pl.count().alias("claim_count"),
    )
    return df
```

This will:
1. Extract all unique lineage IDs from `claims_silver`
2. Create a gold lineage record tracking all parent IDs
3. Support querying which silver records contributed to each gold record

### Example 4: Backfilling with Date Range

```python
@asset(
    io_manager_key="bronze_io_manager",
    metadata={
        "source_file": "claims_backfill_q1_2024.csv",
        "source_system": "sftp",
        "data_date_start": date(2024, 1, 1),
        "data_date_end": date(2024, 3, 31),
        "is_backfill": True,
        "backfill_reason": "Reprocess Q1 data with updated business rules",
        "replaces_lineage_id": "previous-lineage-uuid",  # Optional
    },
)
def claims_bronze_backfill() -> pl.DataFrame:
    df = pl.read_csv("path/to/backfill_file.csv")
    return df
```

This tracks:
- Date range covered by the backfill
- Backfill reason for audit purposes
- Optional link to replaced lineage record

## Lineage Metadata Fields

Assets can provide the following metadata for lineage tracking:

| Field | Type | Description |
|-------|------|-------------|
| `source_file` | str | Source file path or name |
| `source_system` | str | External system identifier (e.g., 'sftp', 'api') |
| `data_date` | date | Single date for the data (for daily partitions) |
| `data_date_start` | date | Start date for date range |
| `data_date_end` | date | End date for date range |
| `is_backfill` | bool | Whether this is a backfill operation |
| `backfill_reason` | str | Explanation for the backfill |
| `replaces_lineage_id` | str | UUID of the lineage record being replaced |
| `parent_lineage_ids` | list[str] | List of UUIDs for parent records (manual override) |
| `transformation_type` | str | Type of transformation (e.g., 'aggregate', 'join', 'filter') |
| `lineage_metadata` | dict | Additional metadata as JSON |

## Querying Lineage

### Find All Source Files for a Dataset

```python
from moncpipelib.lineage import LineageTracker
import sqlalchemy as sa

engine = sa.create_engine("postgresql://...")
tracker = LineageTracker(engine)

records = tracker.query_lineage_history(
    asset_name="claims_silver",
    layer="silver",
    limit=100
)

for record in records:
    print(f"Run ID: {record['run_id']}")
    print(f"Source File: {record['source_file']}")
    print(f"Processed: {record['processed_at']}")
    print(f"Row Count: {record['row_count']}")
```

### Find All Backfills

```python
backfills = tracker.query_lineage_history(
    asset_name="claims_bronze",
    is_backfill=True,
    limit=50
)
```

### Trace Gold Record Back to Bronze Sources

```sql
-- Given a gold record with _lineage_id = 'gold-uuid'
WITH RECURSIVE lineage_tree AS (
    -- Start with the gold record
    SELECT lineage_id, asset_name, layer, source_file, parent_lineage_ids, 1 as depth
    FROM lineage.data_lineage
    WHERE lineage_id = 'gold-uuid'

    UNION ALL

    -- Recursively find parent records
    SELECT dl.lineage_id, dl.asset_name, dl.layer, dl.source_file, dl.parent_lineage_ids, lt.depth + 1
    FROM lineage.data_lineage dl
    JOIN lineage_tree lt ON dl.lineage_id = ANY(lt.parent_lineage_ids)
)
SELECT * FROM lineage_tree
ORDER BY depth, asset_name;
```

### Find Which Silver Records Contributed to a Gold Aggregation

```sql
-- Given gold record with _lineage_id = 'gold-uuid'
SELECT sr.*
FROM synthetic_silver.claims sr
JOIN lineage.data_lineage gold_lineage ON gold_lineage.lineage_id = 'gold-uuid'
WHERE sr._lineage_id = ANY(gold_lineage.parent_lineage_ids);
```

### Find All Records from a Specific Source File

```sql
-- Find all records across all layers from a specific source file
SELECT
    dl.layer,
    dl.asset_name,
    dl.row_count,
    dl.processed_at
FROM lineage.data_lineage dl
WHERE dl.source_file = 'claims_2024_01_15.csv'
ORDER BY dl.layer, dl.asset_name;
```

### Track Backfill History

```sql
-- Find the full backfill chain for a specific data date
WITH RECURSIVE backfill_chain AS (
    -- Start with the latest backfill for a date
    SELECT lineage_id, data_date, is_backfill, backfill_reason, replaces_lineage_id, processed_at, 1 as version
    FROM lineage.data_lineage
    WHERE data_date = '2024-01-15'
      AND is_backfill = FALSE

    UNION ALL

    -- Find previous versions that were replaced
    SELECT dl.lineage_id, dl.data_date, dl.is_backfill, dl.backfill_reason, dl.replaces_lineage_id, dl.processed_at, bc.version + 1
    FROM lineage.data_lineage dl
    JOIN backfill_chain bc ON dl.lineage_id = bc.replaces_lineage_id
)
SELECT * FROM backfill_chain
ORDER BY version DESC;
```

## Benefits

1. **Automatic Tracking**: No manual code needed in assets - lineage is tracked by the IO Manager
2. **Row-Level Granularity**: Track individual records through transformations and aggregations
3. **Backfill Support**: Full history of data reprocessing with reasons
4. **Aggregation Lineage**: Track many:1 relationships through `parent_lineage_ids`
5. **Temporal Tracking**: Support for date partitions and date ranges
6. **Audit Trail**: Complete history of when and why data was processed
7. **Opt-Out Design**: Enabled by default, can be disabled per resource (`enable_row_lineage=False` on `PostgresResource`)

## Resilience & Recovery Features

The lineage system is designed to be resilient to failures and provide maximum debuggability:

### UUID7 Time-Ordering

UUIDs are generated using UUID7 (RFC 9562), which embeds a timestamp in the first 48 bits:

```python
from moncpipelib.lineage import extract_timestamp_from_uuid7
import uuid

# Extract timestamp from any UUID7
lineage_uuid = uuid.UUID("01933b89-1234-7abc-def0-123456789abc")
timestamp = extract_timestamp_from_uuid7(lineage_uuid)
print(f"Created at: {timestamp}")  # 2026-01-14 12:34:56.789 UTC
```

**Benefits**:
- Chronologically sortable without querying the database
- Can determine when data was loaded even if lineage table is lost
- Better database index performance (sequential vs random)
- Easier debugging with timestamp visible in UUID

### Composite Backup Keys

Each lineage record has a `lineage_key` in addition to the UUID:

**Format**: `v{version}:{asset}:{layer}:{date_or_hash}:{run_id_prefix}`

**Examples**:
- `v1:claims:bronze:2024-01-15:abc123` (with data date)
- `v1:claims:silver:f3a2d8e1:abc123` (file hash if no date)
- `v1:claims_summary:gold:20260114123456:abc123` (timestamp fallback)

**Note**: The lineage_key is shared across all rows in a load and is NOT unique. It serves as a human-readable identifier for the load operation.

```sql
-- Find lineage by human-readable key
SELECT * FROM lineage.data_lineage
WHERE lineage_key = 'claims:bronze:2024-01-15:abc123';

-- Pattern matching for all loads on a date
SELECT * FROM lineage.data_lineage
WHERE lineage_key LIKE 'claims:bronze:2024-01-15:%';
```

### Recovery Scenarios

**Scenario 1: Lineage Table Lost**

Even if the lineage table is corrupted or lost, you still have:
1. **UUID7 timestamps** in `_lineage_id` columns on data tables
2. **Run IDs** in `_{layer}_run_id` metadata columns
3. **Processing timestamps** in `_{layer}_processed_at` metadata columns

```sql
-- Reconstruct approximate lineage from data table metadata
SELECT
    _lineage_id as lost_lineage_id,
    _bronze_run_id as run_id,
    to_timestamp(_bronze_processed_at) as processed_at,
    COUNT(*) as row_count
FROM synthetic_bronze.claims
GROUP BY _lineage_id, _bronze_run_id, _bronze_processed_at
ORDER BY processed_at DESC;
```

**Scenario 2: Foreign Key Broken**

If foreign key constraints are removed or data tables are copied:
- Use `lineage_key` pattern matching to correlate records
- Extract timestamps from UUID7s to narrow search window
- Match on `run_id` and `processed_at` ranges

**Scenario 3: Correlating Multiple Loads**

Multiple loads of the same asset/date will have different lineage_ids but the same lineage_key:
```sql
-- Find all loads for a specific asset/date
SELECT lineage_id, lineage_key, processed_at, row_count
FROM lineage.data_lineage
WHERE lineage_key LIKE 'v1:claims:bronze:2024-01-15:%'
ORDER BY processed_at DESC;
```

This is useful for:
- Identifying duplicate loads that need cleanup
- Tracking reprocessing history
- Finding the most recent load for a given date

### Debugging Workflow

When investigating data issues:

1. **Start with the data row**:
   ```sql
   SELECT _lineage_id, _bronze_run_id, _bronze_processed_at
   FROM synthetic_bronze.claims
   WHERE claim_id = '12345';
   ```

2. **Extract timestamp from UUID7** (even if lineage table is down):
   ```python
   from moncpipelib.lineage import extract_timestamp_from_uuid7
   import uuid

   lineage_id = uuid.UUID("01933b89-1234-7abc-def0-123456789abc")
   print(f"Loaded at: {extract_timestamp_from_uuid7(lineage_id)}")
   ```

3. **Query lineage table** for full context:
   ```sql
   SELECT * FROM lineage.data_lineage
   WHERE lineage_id = '01933b89-1234-7abc-def0-123456789abc';
   ```

4. **Use composite key** if UUID lookup fails:
   ```sql
   SELECT * FROM lineage.data_lineage
   WHERE lineage_key LIKE 'claims:bronze:2024-01-15:%';
   ```

## Performance Considerations

- Lineage records are inserted once per asset run (not per row)
- The `_lineage_id` column adds 16 bytes per row (UUID)
- The `lineage_key` column is indexed for fast pattern matching queries
- Foreign key constraints ensure referential integrity
- Indexes on common query patterns (asset_name, data_date, source_file)
- UUID7 provides better index performance than UUID4 (sequential vs random)
- For very high-volume tables, consider disabling lineage on the resource that serves them (`enable_row_lineage=False`)

## Comparison to OpenLineage

| Feature | moncpipelib Lineage | OpenLineage |
|---------|---------------------|-------------|
| Granularity | Row-level | Dataset-level |
| Storage | Same database | External system |
| Query Performance | Fast (SQL joins) | Requires API calls |
| Aggregation Tracking | Native support | Requires custom logic |
| Backfill History | Full versioning | Limited |
| Integration Effort | Automatic (IO Manager) | Manual instrumentation |

Use moncpipelib lineage for row-level tracking within your data platform. Use OpenLineage for cross-system dataset-level lineage.