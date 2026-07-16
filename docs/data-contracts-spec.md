# Data Contracts Specification

This document specifies the data contracts system for moncpipelib, enabling declarative schema and validation rules that generate Dagster asset checks automatically.

## Overview

### Goals

1. Define data expectations declaratively in YAML files
2. Automatically generate Dagster asset checks from contracts
3. Enforce schema and validation rules at materialization time
4. Integrate with OpenLineage for schema metadata emission
5. Provide clear ownership and SLA metadata for data assets

### Non-Goals

- Replace Dagster's native asset check API (contracts generate asset checks)
- Implement a full data quality platform (use Soda/Great Expectations for advanced needs)
- Runtime schema evolution (contracts are static definitions)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Contract YAML Files                                 │
│                 (stored alongside asset definitions)                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   moncpipelib.contracts                                 │
│  ├── loader.py      - Parse YAML, validate structure                    │
│  ├── models.py      - DataContract, Column, Test dataclasses           │
│  ├── checks.py      - Generate Dagster asset checks                     │
│  └── validators.py  - Runtime validation logic                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Dagster Asset Checks                                 │
│  - Schema validation (column names, types)                              │
│  - Column-level tests (not_null, unique, accepted_values, etc.)        │
│  - Table-level expectations (row_count, freshness)                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Dagster UI / Alerts                                  │
│  - Check results visible per asset                                      │
│  - Failed checks block downstream assets (configurable)                 │
│  - Integrates with existing alerting                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Contract File Location

Contracts are stored alongside asset definitions for discoverability:

```
data-platform/
├── assets/
│   ├── bronze/
│   │   ├── claims.py                  # Asset definition
│   │   └── claims.contract.yaml       # Contract
│   ├── silver/
│   │   ├── claims.py
│   │   └── claims.contract.yaml
│   └── gold/
│       ├── claims_summary.py
│       └── claims_summary.contract.yaml
```

Naming convention: `{asset_name}.contract.yaml` in the same directory as the asset.

**Recommended**: Set `contract_search_paths` on the IO manager or pass explicit
`search_paths` to `load_contract_for_asset()`. Implicit discovery via current
working directory or `assets/{layer}/` fallbacks is deprecated.

### Contract Resolution

The `asset` field should be the **bare asset name** without layer prefixes or
slashes (e.g., `fda_ndc_directory`, not `reference_silver/fda_ndc_directory`).
The `layer` field disambiguates when multiple contracts share the same asset name
across different layers:

```yaml
# bronze contract: fda_ndc_directory.contract.yaml
asset: fda_ndc_directory
layer: bronze

# silver contract: fda_ndc_directory_silver.contract.yaml
asset: fda_ndc_directory
layer: silver
```

Contract identity is **sink-qualified** (#405): in addition to the bare `asset`
name, every table sink that declares both `schema` and `table` registers the
contract under the identity `"schema/table"`. Two contracts may therefore share
the same `asset` name and the same `layer` as long as their sinks land in
different schemas (e.g. `synthetic_gold.dim_provider` and
`reference_gold.dim_provider`):

```yaml
# synthetic contract
asset: dim_provider
layer: gold
sinks:
  - type: table
    schema: synthetic_gold
    table: dim_provider

# reference contract -- same asset, same layer, distinct sink
asset: dim_provider
layer: gold
sinks:
  - type: table
    schema: reference_gold
    table: dim_provider
```

**Lookup behavior** (via `load_contract_for_asset()`):

1. **Exact match**: the `asset_name` argument is matched directly against the
   `asset` field in each contract.
2. **Sink-qualified match**: when the `asset_name` contains `/` (e.g., from
   Dagster's `AssetKey.to_user_string()` on repo-convention keys like
   `AssetKey(["reference_gold", "dim_provider"])`), it is matched against the
   sink identities (`"schema/table"`). Keys with extra leading components are
   also tried on their last two components.
3. **Last-component fallback** (legacy): the portion after the last `/` is
   matched against the `asset` field.
4. **Layer disambiguation**: when multiple contracts match the same asset name,
   the `layer` parameter selects the correct one.

**Duplicate handling**: a same-asset/same-layer duplicate that cannot be
disambiguated by sink (overlapping sink identities, or a contract lacking a
schema-qualified table sink) raises `ContractValidationError` when the contract
index is built -- it is never silently dropped. The one exception is
**search-path priority**: when the duplicate comes from a *later* entry in
`search_paths`, the earlier path's contract wins and the later one is shadowed
with a warning (override semantics). A bare-name lookup that still matches
multiple same-layer contracts raises instead of guessing; use the
sink-qualified name to resolve it.

**Write-time sink matching**: when the write target carries a schema (the
resource path's `target="schema.table"`, or `target_schema` asset metadata on
the IO manager path), a sink declaring a *different* schema is never applied --
the mismatch is logged and, for schema resolution, the metadata schema wins.

**Contract checks**: `make_contract_checks()` and `discover_contract_checks()`
recursively scan subdirectories for `*.contract.yaml` files, so a single call
pointed at a top-level `defs/` directory discovers all contracts across layers.

## Contract YAML Schema

### Full Example

```yaml
# claims.contract.yaml
version: "1.0"
pipeline_id: "550e8400-e29b-41d4-a716-446655440000"
asset: claims_bronze
layer: bronze
description: |
  Raw claims data ingested from SFTP.
  Contains one row per claim submission.

owner:
  team: data-engineering
  contact: data-platform@modeloncology.com
  slack_channel: "#data-platform-alerts"

schema:
  # Fail if DataFrame has columns not listed here (default: true)
  strict: true

  columns:
    - name: claim_id
      type: string
      nullable: false
      primary_key: true
      pii: false
      description: Unique claim identifier from source system
      tests:
        - not_null
        - unique

    - name: patient_id
      type: string
      nullable: false
      pii: true  # Masked in logs, tracked in catalog
      phi: true  # Also PHI -- identifies a patient in a clinical context
      description: Patient identifier (FK to patients table)
      tests:
        - not_null
        - pattern: "^PAT-[0-9]{8}$"

    - name: provider_id
      type: string
      nullable: false
      pii: true   # Identifies a person (the provider) ...
      phi: false  # ... but is a business identifier, not PHI
      tests:
        - not_null

    - name: amount
      type: decimal
      nullable: true
      pii: false
      description: Claim amount in USD
      tests:
        - greater_than: 0
          severity: warn
        - less_than: 1000000
          severity: error

    - name: claim_date
      type: date
      nullable: false
      pii: false
      tests:
        - not_null
        - not_in_future
        - within_days: 365

    - name: status
      type: string
      nullable: false
      pii: false
      tests:
        - not_null
        - accepted_values:
            values: ["pending", "approved", "denied", "appealed"]
          severity: error  # modifier: sibling of the test type, not nested

    - name: diagnosis_code
      type: string
      nullable: true
      pii: false
      tests:
        - pattern: "^[A-Z][0-9]{2}(\\.[0-9]{1,2})?$"
          when: not_null
          severity: warn

    - name: _lineage_id
      type: uuid
      nullable: false
      description: moncpipelib lineage tracking (auto-added)
      managed: true  # Indicates this column is managed by moncpipelib

    - name: _lineage_key
      type: string
      nullable: false
      description: moncpipelib lineage key (auto-added)
      managed: true

expectations:
  # NOTE: severity is a sibling of the expectation type, never nested inside
  # its parameter mapping. A nested severity is rejected at load time (#394).
  - row_count:
      min: 1
      max: 10000000
    severity: error

  - freshness:
      column: claim_date
      max_age_hours: 48
    severity: warn

  - null_percentage:
      column: diagnosis_code
      max_percent: 10
    severity: warn

  - unique_combination:
      columns: [claim_id, claim_date]
    severity: error

upstream:
  - name: sftp_claims_file
    type: external
    system: sftp
    description: Daily claims file from clearinghouse

sla:
  freshness_hours: 24
  update_frequency: daily
  availability_percent: 99.9
```

### Schema Reference

#### Root Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | Yes | Contract schema version (currently "1.0") |
| `pipeline_id` | string (UUID) | Yes | Stable UUID identifying the logical pipeline. Persists across asset renames so lineage history remains correlated. |
| `asset` | string | Yes | Bare asset name (no slashes or layer prefixes). Disambiguated by `layer` when the same name exists across layers, and by sink identity (`schema` + `table`) when it exists across schemas within a layer (#405). See [Contract Resolution](#contract-resolution). |
| `layer` | string | Yes | Data layer (bronze/silver/gold) |
| `description` | string | No | Human-readable description |
| `owner` | object | No | Ownership metadata |
| `schema` | object | Yes | Column definitions and settings |
| `expectations` | list | No | Table-level validation rules |
| `upstream` | list | No | Upstream dependencies documentation |
| `sla` | object | No | Service level agreement metadata |
| `lineage` | object | No | Row-level lineage configuration |
| `tags` | dict[str, str] | No | User-defined string tags for Dagster job/op tagging. Keys starting with `moncpipelib/` are reserved for auto-derived tags. |

> **Breaking Change:** The `pipeline_id` field is now required on all contracts.
> Existing contracts that omit `pipeline_id` will fail validation. Update every
> `.contract.yaml` file to include a stable UUID before upgrading.

#### Owner Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `team` | string | Yes | Owning team name |
| `contact` | string | No | Email contact |
| `slack_channel` | string | No | Slack channel for alerts |

#### Schema Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `strict` | boolean | No | If true, fail on unexpected columns (default: true) |
| `columns` | list | Yes | Column definitions |

#### Column Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Column name |
| `type` | string | Yes | Data type (see types below) |
| `nullable` | boolean | Yes | Whether nulls are allowed |
| `primary_key` | boolean | No | Part of primary key (default: false) |
| `description` | string | No | Column description |
| `managed` | boolean | No | Auto-managed by moncpipelib (default: false) |
| `pii` | boolean | No | Whether column contains PII (default: **true** -- safe by default) |
| `phi` | boolean | No | Whether column contains PHI under HIPAA (default: **the `pii` value** -- safe by default) |
| `tests` | list | No | Column-level validation tests |

##### PII vs PHI

`pii` and `phi` are distinct classifications and both are synced per write to
`lineage.column_metadata` tags (`{"pii": <bool>, "phi": <bool>}`) and to the
OpenLineage `ColumnClassificationFacet`:

- **PII** (personally identifiable information): any value that identifies a
  person -- patient, provider, or employee.
- **PHI** (protected health information): individually identifiable health
  information as defined by HIPAA. They diverge in both directions:
  provider NPIs or business contact details are PII but not PHI, while
  properly de-identified clinical values are neither.

When `phi` is not annotated it defaults to the column's `pii` value, so
existing contracts remain valid and every unreviewed column stays PHI-suspect.
Setting `phi: false` is an affirmative, reviewed statement that the column is
safe to read in a HIPAA context -- downstream PHI gates treat `phi: false` as
"cleared" and anything else (true or absent) as protected. A contract whose
non-managed columns are all `phi: false` derives `data_classification: none`
instead of `PHI` in the pipeline registry.

#### Supported Types

| Type | Polars Type | PostgreSQL Type | Description |
|------|-------------|-----------------|-------------|
| `string` | `Utf8` | `TEXT` | Variable-length text |
| `integer` | `Int64` | `BIGINT` | 64-bit integer |
| `decimal` | `Float64` | `DOUBLE PRECISION` | Floating point |
| `boolean` | `Boolean` | `BOOLEAN` | True/false |
| `date` | `Date` | `DATE` | Date without time |
| `datetime` | `Datetime` | `TIMESTAMP` | Date with time |
| `uuid` | `Utf8` | `UUID` | UUID string |
| `json` | `Utf8` | `JSON` | JSON text |
| `jsonb` | `Utf8` | `JSONB` | Binary JSON |

#### Column Tests

| Test | Parameters | Description |
|------|------------|-------------|
| `not_null` | - | Column must not contain nulls |
| `unique` | - | All values must be unique |
| `accepted_values` | `values: list` | Value must be in list |
| `not_in` | `values: list` | Value must NOT be in list |
| `pattern` | `value: string` (or scalar shorthand: `pattern: "^..."`) | Value must match regex |
| `greater_than` | `value: number` | Value must be > threshold |
| `greater_than_or_equal` | `value: number` | Value must be >= threshold |
| `less_than` | `value: number` | Value must be < threshold |
| `less_than_or_equal` | `value: number` | Value must be <= threshold |
| `between` | `min, max: number` | Value must be in range |
| `not_in_future` | - | Date/datetime must not be in future |
| `within_days` | `days: int` | Date must be within N days of today |
| `min_length` | `length: int` | String must be at least N chars |
| `max_length` | `length: int` | String must be at most N chars |

#### Test Modifiers

All tests support these optional modifiers:

| Modifier | Type | Default | Description |
|----------|------|---------|-------------|
| `severity` | string | `error` | `error` (blocks downstream) or `warn` (logs only) |
| `when` | string | - | Condition: `not_null` (only test non-null values) |

Modifiers are **siblings of the test type key**, never nested inside its
parameter mapping:

```yaml
tests:
  - between:
      min: 0
      max: 100
    severity: warn   # correct: sibling of `between`
```

A modifier nested inside the parameter mapping would land in the test's
`parameters` dict and be silently ignored at runtime, so
`validate_contract_schema` rejects it at load time, along with any parameter
key not valid for the test type (#394).

#### Table Expectations

| Expectation | Parameters | Description |
|-------------|------------|-------------|
| `row_count` | `min, max: int` | Table row count bounds |
| `freshness` | `column: string, max_age_hours: int` | Most recent value age |
| `null_percentage` | `column: string, max_percent: float` | Max % of nulls allowed |
| `unique_combination` | `columns: list` | Composite uniqueness |
| `history_completeness` | - | Post-write check that all contract periods are represented |

The same placement rule applies: `severity` is a sibling of the expectation
type key (see the Full Example above). A `severity` nested inside the
parameter mapping -- or any parameter key not listed for the expectation
type -- is a load-time validation error (#394). `when` is not supported on
table expectations.

#### Lineage Object

Controls row-level lineage tracking for this asset. When the `lineage` section is
omitted, lineage uses default behavior: enabled with no `source_system` or
`transformation_type`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | boolean | `true` | Whether row-level lineage tracking is enabled. Set `false` to skip lineage even when the resource has `enable_row_lineage=True`. |
| `source_system` | string | - | External system identifier (e.g., `"openfda"`, `"sftp"`, `"api"`). Stored in the lineage record for provenance. |
| `transformation_type` | string | - | Type of transformation applied (e.g., `"ingest"`, `"aggregate"`, `"join"`, `"filter"`). Stored in the lineage record. |

**Layer resolution for lineage:** The contract's `layer` field (bronze/silver/gold) is
used to resolve the data layer when the target schema does not directly match a valid
layer name. For example, writing to `reference_bronze.fda_ndc_directory` with a contract
declaring `layer: bronze` will correctly enable lineage with layer `bronze`.

**Example:**

```yaml
lineage:
  source_system: openfda
  transformation_type: ingest
```

**Disabling lineage for a specific asset:**

```yaml
lineage:
  enabled: false
```

#### Tags

The optional `tags` section defines string key-value pairs that can be loaded
into Dagster job or asset tags via `ContractTags.from_contract()`. All keys
and values must be strings. Keys starting with `moncpipelib/` are reserved for
auto-derived tags (layer, owner, has_sla, etc.) and should not be used in
contract YAML.

```yaml
tags:
  team/priority: high
  oncall/pager: "true"
  data_classification: internal
```

#### Sources and Sinks

The `sources` list documents where an asset reads data from. The `sinks` list
declares where it writes data to. The IO manager reconciles sink declarations
against asset metadata at write time.

**Table sink fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Must be `"table"` |
| `schema` | string | Yes | Target database schema |
| `table` | string | Yes | Target table name |
| `database` | string | No | Target database (default: `analytics`) |
| `description` | string | No | Human-readable description |
| `mode` | string | No | Write mode: `full_refresh`, `upsert`, `append`, `scd2` |
| `business_key` | list[string] or string | No | SCD2: stable identifier columns |
| `tracked_columns` | list[string] or string | No | SCD2: columns whose changes trigger a new version. An explicit empty list selects presence-only SCD2 (see below). |
| `detect_deletes` | boolean | No | SCD2: expire absent records (default: false) |
| `sequence_column` | string or null | No | SCD2: per-business-key version counter column (default: `seq_id`). Set to `null` to opt out. |
| `partition_column` | string | No | Column mapping Dagster partition key to SQL column |
| `primary_key` | list[string] or string | No | Upsert conflict-key columns. When declared, takes precedence over schema-level `primary_key: true` columns for write configuration (the two may legitimately differ: sink-level names the upsert conflict key, schema-level often marks a surrogate identifier). |
| `skip_unchanged` | boolean | No | Upsert: suppress `DO UPDATE` for conflicting rows whose update columns are all unchanged (default: false). See the upsert change-guard note below. |

**Upsert change-guard (`skip_unchanged`):**

```yaml
sinks:
  - type: table
    schema: silver
    table: dim_catalog
    mode: upsert
    primary_key: [catalog_id]
    skip_unchanged: true
```

With `skip_unchanged: true` the upsert merge adds a NULL-safe
`WHERE target.col IS DISTINCT FROM EXCLUDED.col OR ...` guard over the update
columns, so a re-loaded row whose values did not change produces no heap
write, no index churn, and no WAL -- valuable for high-cardinality, low-churn
loads (mirror issue model-oncology-public/moncpipelib#3). It is opt-in
because it changes observable side effects: row-level `ON UPDATE` triggers
(e.g. `updated_at` touch triggers) no longer fire for unchanged rows, and any
consumer relying on "every load touches every row" needs the default.
Conflicting rows are still locked while the guard is evaluated, so
concurrency semantics are unchanged. The `rows_upserted` statistic still
reports the incoming row count.

Two practical caveats:

- Every update column's type must have an equality operator. `json` (unlike
  `jsonb`), `xml`, and some geometric types do not, and the guarded merge
  then fails with "could not identify an equality operator" even though the
  default (unguarded) upsert works. On such tables, scope `update_columns`
  to exclude those columns or leave the guard off.
- The caller-supplied side of the four-way reconciliation for this field is
  the resource `write(..., skip_unchanged=...)` keyword argument. The retired
  IO-manager path does not accept a `skip_unchanged` asset-metadata key;
  contract sinks work on both paths.

**SCD2 sink example:**

```yaml
sinks:
  - type: table
    schema: silver
    table: dim_product
    mode: scd2
    business_key: [product_id]
    tracked_columns: [name, price]
    detect_deletes: false
    sequence_column: seq_id  # default; set to null to opt out
```

**Presence-only SCD2 (`tracked_columns: []`, #432):**

```yaml
sinks:
  - type: table
    schema: reference_silver
    table: npi_other_name
    mode: scd2
    business_key: [npi, other_organization_name, other_organization_name_type_code, created_date]
    tracked_columns: []   # presence-only: the key IS the full tuple
    detect_deletes: true
```

Junction/reference tables whose business key is the full source tuple have no
non-key attributes to change-detect. Declaring `tracked_columns: []` (or
omitting it when the business key covers every DataFrame column) selects
presence-only versioning: `row_hash` is computed over the sorted business key,
so the writer's change predicate never fires within a key. A tuple's
appearance in a load opens an `effective_from` span; its absence (via
`detect_deletes: true`) closes it; reappearance opens a new span. Pair this
mode with `detect_deletes: true` -- without it nothing ever expires, and the
write logs a warning to that effect.

Guard rails are unchanged: a business key that does not uniquely identify
rows per partition is still rejected before any DML by the staging
uniqueness check (#419), and SCD2 without any `business_key` remains an
error.

Migration note: converting an existing SCD2 table from a formal
single-column workaround (e.g. `tracked_columns: [created_date]` where
`created_date` is also a key column) to `tracked_columns: []` changes the
`row_hash` formula. On the first write after the switch, every existing
current row hashes differently and is expired-and-reinserted once (a
spurious full version churn, not corruption). To avoid it, backfill
`row_hash` on current rows to the new formula (SHA-256 over the sorted
business-key columns joined with `|`, nulls as `\x00`) before the first
presence-only write.

**SCD2 bookkeeping columns and `schema.columns`:** the bookkeeping columns
(`effective_from`, `effective_to`, `is_current`, `row_hash`, and the optional
`sequence_column`) are managed at the target-table level by the SCD2 writer --
they never appear in the DataFrame the asset returns, and the cookbook examples
do not declare them in `schema.columns`. Declaring them anyway with
`managed: true` is **permitted and harmless**: managed columns are excluded
from DataFrame schema validation (including strict mode), column-level tests,
and the PII/PHI classification rollup, so both styles load and write
identically. Choose whichever your team prefers for documentation value;
just never declare them without `managed: true`, since strict validation
would then expect them in the DataFrame.

**Reconciliation rules at write time:**

All sink fields that correspond to IO manager configuration (`mode`, `business_key`,
`detect_deletes`, `sequence_column`, `partition_column`, `primary_key`,
`skip_unchanged`) follow the same four-way rule:

| Contract | Asset Metadata | Result |
|----------|----------------|--------|
| Set | Not set | Contract value used silently |
| Not set | Set | Metadata value used unchanged |
| Set | Set (same value) | Warning logged, write proceeds |
| Set | Set (different value) | `ContractViolationError` — always fatal |

`tracked_columns` has a stricter rule: because `tracked_columns` absent from metadata
means "hash all non-BK columns" (a specific behaviour), a contract declaring specific
columns is always treated as a conflict if metadata does not also explicitly set
`tracked_columns` to the same list.

**Load-time validation (static guard rails):**

Beyond structural checks, `validate_contract_schema` / `load_contract` reject
statically-detectable write hazards at contract load so misconfiguration costs
seconds at import instead of a dead or silently-corrupting pipeline at runtime
(#401):

- A column declaring both `primary_key: true` and `nullable: true` is
  rejected: NULL keys bypass upsert `ON CONFLICT` matching, so NULL-keyed rows
  silently duplicate on every run. The same rule applies to sink-level
  `primary_key` members whose schema column is nullable (upsert sinks).
- An upsert sink declaring `partition_column` whose effective primary key does
  not include that column is rejected (the upsert would match records across
  partitions).
- `detect_deletes: true` on a sink whose declared `mode` is not `scd2` is
  rejected (the flag is consumed only by the SCD2 write path).
- `skip_unchanged: true` on a sink whose declared `mode` is not `upsert` is
  rejected for the same reason (the flag is consumed only by the upsert
  merge).
- Both mode-scoped flags are also backstopped at write time:
  `validate_write_config` rejects `detect_deletes=True` outside `scd2` and
  `skip_unchanged=True` outside `upsert`, covering configuration whose mode
  arrives via asset metadata or the `write()` call rather than a declared
  sink mode.
- Sink `primary_key` members must be declared in `schema.columns`
  (`business_key` and `tracked_columns` were already cross-referenced;
  `partition_column` is deliberately exempt because it is injected into the
  DataFrame at write time and is conventionally absent from the schema block).
- When the contract's resolved `data_source` declares partitioning (any period
  with a `partition_key`, or `periods.mode: from_ingest`), a table sink with
  `mode: full_refresh` or `mode: scd2` must declare `partition_column`. This is
  the static form of the write-time partition guard in
  `validate_partition_safety`, which cannot fire if the run dies before
  reaching `database.write()`.

The write-time guards remain as backstops for configuration supplied via asset
metadata rather than contract YAML. The upsert writer additionally rejects
DataFrames carrying NULLs in any `primary_key` column at write time.

## Module Structure

```
src/moncpipelib/
├── contracts/
│   ├── __init__.py          # Public API exports
│   ├── models.py            # DataContract, Column, Test dataclasses
│   ├── loader.py            # YAML parsing and validation
│   ├── checks.py            # Dagster asset check generation
│   └── validators.py        # Runtime validation logic
```

## API Specification

### models.py

```python
from dataclasses import dataclass, field
from enum import Enum

class Severity(str, Enum):
    ERROR = "error"
    WARN = "warn"

class ColumnType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    UUID = "uuid"
    JSON = "json"
    JSONB = "jsonb"

@dataclass
class ColumnTest:
    """A validation test for a column."""
    test_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    severity: Severity = Severity.ERROR
    when: str | None = None

@dataclass
class Column:
    """Column definition in a data contract."""
    name: str
    type: ColumnType
    nullable: bool
    description: str | None = None
    primary_key: bool = False
    managed: bool = False
    tests: list[ColumnTest] = field(default_factory=list)

@dataclass
class TableExpectation:
    """Table-level validation expectation."""
    expectation_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    severity: Severity = Severity.ERROR

@dataclass
class Owner:
    """Ownership metadata."""
    team: str
    contact: str | None = None
    slack_channel: str | None = None

@dataclass
class UpstreamDependency:
    """Documentation of upstream data source."""
    name: str
    type: str  # "asset" or "external"
    system: str | None = None
    description: str | None = None

@dataclass
class SLA:
    """Service level agreement metadata."""
    freshness_hours: int | None = None
    update_frequency: str | None = None
    availability_percent: float | None = None

@dataclass
class Schema:
    """Schema definition."""
    columns: list[Column]
    strict: bool = True

@dataclass
class DataContract:
    """Complete data contract definition."""
    version: str
    pipeline_id: str
    asset: str
    layer: str
    schema: Schema
    description: str | None = None
    owner: Owner | None = None
    expectations: list[TableExpectation] = field(default_factory=list)
    upstream: list[UpstreamDependency] = field(default_factory=list)
    sla: SLA | None = None

    def get_primary_key_columns(self) -> list[str]:
        """Return list of primary key column names."""
        return [c.name for c in self.schema.columns if c.primary_key]

    def get_column(self, name: str) -> Column | None:
        """Get column definition by name."""
        for col in self.schema.columns:
            if col.name == name:
                return col
        return None
```

### loader.py

```python
from pathlib import Path
import yaml

class ContractValidationError(Exception):
    """Raised when contract YAML is invalid."""
    pass

class ContractNotFoundError(Exception):
    """Raised when contract file doesn't exist."""
    pass

def load_contract(path: str | Path) -> DataContract:
    """Load and validate a contract from a YAML file.

    Args:
        path: Path to the contract YAML file

    Returns:
        DataContract: Parsed and validated contract

    Raises:
        ContractNotFoundError: If file doesn't exist
        ContractValidationError: If YAML structure is invalid
    """
    ...

def load_contract_for_asset(
    asset_name: str,
    layer: str | None = None,
    search_paths: list[Path | str] | None = None,
    caller_file: str | None = None,
) -> DataContract | None:
    """Find and load contract for an asset by convention.

    Searches for {asset_name}.contract.yaml in:
    1. Provided search_paths (RECOMMENDED)
    2. Current working directory (DEPRECATED)
    3. assets/{layer}/ directory (DEPRECATED)

    Pass explicit search_paths to ensure deterministic contract
    resolution across all environments (tests, CI, async).

    Args:
        asset_name: Name of the asset
        layer: Data layer (bronze/silver/gold), optional
        search_paths: Explicit directories to search (recommended)
        caller_file: Path to the calling file (auto-detected if None)

    Returns:
        DataContract if found, None otherwise
    """
    ...

def validate_contract_schema(data: dict) -> list[str]:
    """Validate contract YAML structure.

    Args:
        data: Parsed YAML dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    ...
```

### checks.py

```python
from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    AssetChecksDefinition,
    asset_check,
)

def generate_asset_checks(
    contract: DataContract,
    blocking: bool = True,
) -> list[AssetChecksDefinition]:
    """Generate Dagster asset checks from a data contract.

    Creates checks for:
    - Schema validation (column presence, types)
    - Column-level tests (not_null, unique, etc.)
    - Table-level expectations (row_count, freshness, etc.)

    Args:
        contract: The data contract to generate checks from
        blocking: If True, ERROR severity checks block downstream assets

    Returns:
        List of Dagster AssetChecksDefinition objects
    """
    ...

def generate_schema_check(contract: DataContract) -> AssetChecksDefinition:
    """Generate a schema validation check.

    Validates:
    - All required columns are present
    - Column types match expected types
    - No unexpected columns (if strict mode enabled)

    Args:
        contract: The data contract

    Returns:
        AssetChecksDefinition for schema validation
    """
    ...

def generate_column_checks(contract: DataContract) -> list[AssetChecksDefinition]:
    """Generate column-level validation checks.

    Creates one check per column test defined in the contract.

    Args:
        contract: The data contract

    Returns:
        List of AssetChecksDefinition for column tests
    """
    ...

def generate_expectation_checks(contract: DataContract) -> list[AssetChecksDefinition]:
    """Generate table-level expectation checks.

    Creates checks for row_count, freshness, null_percentage, etc.

    Args:
        contract: The data contract

    Returns:
        List of AssetChecksDefinition for table expectations
    """
    ...

# Convenience function for loading + generating in one call
def load_contract_checks(
    asset_name: str,
    layer: str,
    search_paths: list[Path] | None = None,
    blocking: bool = True,
) -> list[AssetChecksDefinition]:
    """Load contract and generate asset checks.

    Convenience function combining load_contract_for_asset
    and generate_asset_checks.

    Args:
        asset_name: Name of the asset
        layer: Data layer
        search_paths: Optional search paths for contract file
        blocking: If True, ERROR severity checks block downstream

    Returns:
        List of asset checks, or empty list if no contract found
    """
    contract = load_contract_for_asset(asset_name, layer, search_paths)
    if contract is None:
        return []
    return generate_asset_checks(contract, blocking)
```

### validators.py

```python
import polars as pl

@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    message: str
    failed_count: int = 0
    total_count: int = 0
    sample_failures: list[dict] | None = None

def validate_schema(
    df: pl.DataFrame,
    contract: DataContract,
) -> ValidationResult:
    """Validate DataFrame schema against contract.

    Checks:
    - All non-managed columns in contract exist in DataFrame
    - Column types match (with type coercion tolerance)
    - No unexpected columns (if strict mode)

    Args:
        df: DataFrame to validate
        contract: Contract defining expected schema

    Returns:
        ValidationResult with pass/fail and details
    """
    ...

def validate_not_null(
    df: pl.DataFrame,
    column: str,
) -> ValidationResult:
    """Check column contains no null values."""
    null_count = df.filter(pl.col(column).is_null()).height
    return ValidationResult(
        passed=null_count == 0,
        message=f"Column '{column}' has {null_count} null values",
        failed_count=null_count,
        total_count=df.height,
    )

def validate_unique(
    df: pl.DataFrame,
    column: str,
) -> ValidationResult:
    """Check column contains only unique values."""
    ...

def validate_accepted_values(
    df: pl.DataFrame,
    column: str,
    values: list[Any],
) -> ValidationResult:
    """Check column values are in accepted list."""
    ...

def validate_pattern(
    df: pl.DataFrame,
    column: str,
    pattern: str,
) -> ValidationResult:
    """Check column values match regex pattern."""
    ...

def validate_greater_than(
    df: pl.DataFrame,
    column: str,
    threshold: float,
) -> ValidationResult:
    """Check column values are greater than threshold."""
    ...

def validate_less_than(
    df: pl.DataFrame,
    column: str,
    threshold: float,
) -> ValidationResult:
    """Check column values are less than threshold."""
    ...

def validate_not_in_future(
    df: pl.DataFrame,
    column: str,
) -> ValidationResult:
    """Check date/datetime column has no future values."""
    ...

def validate_within_days(
    df: pl.DataFrame,
    column: str,
    days: int,
) -> ValidationResult:
    """Check date column values are within N days of today."""
    ...

def validate_row_count(
    df: pl.DataFrame,
    min_count: int | None = None,
    max_count: int | None = None,
) -> ValidationResult:
    """Check table row count is within bounds."""
    ...

def validate_freshness(
    df: pl.DataFrame,
    column: str,
    max_age_hours: int,
) -> ValidationResult:
    """Check most recent value in column is within max age."""
    ...

def validate_null_percentage(
    df: pl.DataFrame,
    column: str,
    max_percent: float,
) -> ValidationResult:
    """Check null percentage in column is below threshold."""
    ...

def validate_unique_combination(
    df: pl.DataFrame,
    columns: list[str],
) -> ValidationResult:
    """Check combination of columns is unique."""
    ...
```

## Usage Examples

### Basic Usage

```python
from dagster import asset, Definitions
from moncpipelib import PostgresIOManager, load_contract_checks
import polars as pl

@asset(io_manager_key="bronze_io_manager")
def claims_bronze() -> pl.DataFrame:
    # Load data from source
    return pl.read_csv("claims.csv")

# Generate checks from contract
claims_checks = load_contract_checks("claims_bronze", "bronze")

defs = Definitions(
    assets=[claims_bronze],
    asset_checks=claims_checks,
    resources={
        "bronze_io_manager": PostgresIOManager(...),
    },
)
```

### Decorator Pattern (Alternative)

```python
from moncpipelib.contracts import with_contract

@with_contract  # Auto-discovers claims_bronze.contract.yaml
@asset(io_manager_key="bronze_io_manager")
def claims_bronze() -> pl.DataFrame:
    return pl.read_csv("claims.csv")

# Checks are automatically registered
```

### Manual Contract Loading

```python
from moncpipelib.contracts import load_contract, generate_asset_checks

# Load from specific path
contract = load_contract("path/to/claims.contract.yaml")

# Generate checks with custom settings
checks = generate_asset_checks(contract, blocking=False)

# Access contract metadata
print(f"Owner: {contract.owner.team}")
print(f"Primary key: {contract.get_primary_key_columns()}")
```

### Runtime Validation (Outside Dagster)

```python
from moncpipelib.contracts import load_contract, validate_schema, validate_not_null

contract = load_contract("claims.contract.yaml")
df = pl.read_csv("claims.csv")

# Validate schema
result = validate_schema(df, contract)
if not result.passed:
    print(f"Schema validation failed: {result.message}")

# Validate specific column
result = validate_not_null(df, "claim_id")
if not result.passed:
    print(f"Found {result.failed_count} null values in claim_id")
```

## Integration with OpenLineage

When emitting OpenLineage events, the contract's schema is used to populate the `SchemaDatasetFacet`:

```python
# In PostgresIOManager or OpenLineageEmitter

def _build_schema_facet(
    self,
    df: pl.DataFrame,
    contract: DataContract | None,
) -> SchemaDatasetFacet:
    """Build schema facet from contract or DataFrame inference.

    Contract takes precedence as source of truth for:
    - Column descriptions
    - Expected types (vs actual types)
    - Primary key information
    """
    if contract:
        return SchemaDatasetFacet(
            fields=[
                SchemaDatasetFacetFields(
                    name=col.name,
                    type=col.type.value,
                    description=col.description,
                )
                for col in contract.schema.columns
                if not col.managed  # Exclude moncpipelib-managed columns
            ]
        )
    else:
        # Fall back to DataFrame inference
        return self._infer_schema_from_df(df)
```

The contract's `owner` and `sla` fields can also populate OpenLineage facets:

```python
# OwnershipJobFacet from contract.owner
# Custom SLAFacet from contract.sla
```

## Integration with PostgresIOManager

Optional schema enforcement at write time:

```python
class PostgresIOManager(ConfigurableIOManager):
    # ... existing fields ...

    enforce_contracts: bool = False
    """If True, validate against contract before writing. Default: False
    (prefer asset checks for validation, but this provides a hard stop)."""

    contract_search_paths: list[str] | None = None
    """Paths to search for contract files."""

    def handle_output(self, context: OutputContext, obj: pl.DataFrame) -> None:
        # Load contract if exists
        contract = load_contract_for_asset(
            context.asset_key.to_user_string(),
            self.layer,
            self.contract_search_paths,
        )

        # Optional: enforce schema at write time (in addition to asset checks)
        if contract and self.enforce_contracts:
            result = validate_schema(obj, contract)
            if not result.passed:
                raise ContractViolationError(
                    f"Schema validation failed for {context.asset_key}: {result.message}"
                )

        # Continue with existing logic...
        lineage_id, lineage_key = self._add_lineage(obj, context, source_file)

        # Pass contract to OpenLineage emitter for accurate schema
        if self.openlineage_config.enabled:
            emitter.emit_run_event(
                ...,
                contract=contract,
            )
```

## Generated Check Naming

Asset checks are named systematically for clarity in Dagster UI:

| Check Type | Name Pattern | Example |
|------------|--------------|---------|
| Schema | `{asset}_schema` | `claims_bronze_schema` |
| Column not_null | `{asset}_{column}_not_null` | `claims_bronze_claim_id_not_null` |
| Column unique | `{asset}_{column}_unique` | `claims_bronze_claim_id_unique` |
| Column pattern | `{asset}_{column}_pattern` | `claims_bronze_patient_id_pattern` |
| Column accepted_values | `{asset}_{column}_accepted_values` | `claims_bronze_status_accepted_values` |
| Row count | `{asset}_row_count` | `claims_bronze_row_count` |
| Freshness | `{asset}_freshness` | `claims_bronze_freshness` |

## Validation Surfaces and SCD2 Semantics

Contract rules execute on two distinct surfaces with different data in
scope. Contract authors -- especially for `mode: scd2` sinks -- should
understand which rows each surface validates (issue #418).

### Write-path validation (PostgresResource / PostgresIOManager)

Column tests run against the **incoming DataFrame only**, before any
merge or SCD2 staging. `unique: true` on an SCD2 business key therefore
means "unique within this snapshot" and is a valid, useful test: the
SCD2 writer requires one row per business key per write (unless a
sequence column is used for multi-version historical loads), and the
in-frame check catches genuinely duplicated incoming data -- for
example, an upstream read that picked up double-current rows.

For batched writes, contract enforcement runs on the first batch only;
cross-batch duplicates are not detected on this surface.

### Asset-check validation (make_contract_checks and related)

Checks execute against the **sink table** (SQL pushdown or full-table
Polars load). For sinks with `mode: scd2`, all column tests and table
expectations are automatically scoped to **current rows**
(`is_current = TRUE`): an SCD2 table legitimately repeats business keys
across history rows, so an unscoped `unique` check would fail
deterministically after the first change wave, and other test types
(`accepted_values`, `within_days`, ...) would degrade as expired
history accrues. Scoped checks validate the table "as of now", matching
the write-path snapshot semantics.

Notes:

- The current-flag column name comes from the writer default
  (`SCD2Config.is_current_col`, i.e. `is_current`); sinks cannot
  override SCD2 column names in contract YAML.
- The scope follows the contract's **first table sink** -- the same
  sink that determines which table the checks run against.
- Check result metadata carries `scope: current rows only
  (is_current = TRUE)` when scoping is active.
- Schema validation is unaffected by row scoping (it inspects columns
  and types, not rows).
- If a `mode: scd2` sink's table lacks the current-flag column, the
  Polars path raises instead of silently validating full history; the
  SQL path fails with an undefined-column error.

## Check Severity Mapping

| Contract Severity | Dagster Severity | Behavior |
|-------------------|------------------|----------|
| `error` | `AssetCheckSeverity.ERROR` | Blocks downstream assets |
| `warn` | `AssetCheckSeverity.WARN` | Logs warning, allows downstream |

## Testing

### Unit Tests

Location: `tests/test_contracts.py`

```python
def test_load_contract_parses_valid_yaml():
    """Verify valid contract YAML is parsed correctly."""
    ...

def test_load_contract_raises_on_invalid_yaml():
    """Verify invalid YAML raises ContractValidationError."""
    ...

def test_generate_asset_checks_creates_schema_check():
    """Verify schema check is generated."""
    ...

def test_generate_asset_checks_creates_column_checks():
    """Verify column tests become asset checks."""
    ...

def test_validate_not_null_detects_nulls():
    """Verify not_null validation catches null values."""
    ...

def test_validate_pattern_matches_regex():
    """Verify pattern validation works with regex."""
    ...

def test_severity_mapping_blocks_on_error():
    """Verify ERROR severity blocks downstream."""
    ...

def test_severity_mapping_warns_on_warn():
    """Verify WARN severity allows downstream."""
    ...
```

### Integration Tests

```python
@pytest.mark.integration
def test_contract_checks_run_in_dagster():
    """Test generated checks execute correctly in Dagster."""
    ...

@pytest.mark.integration
def test_contract_blocks_downstream_on_failure():
    """Test ERROR severity checks block downstream assets."""
    ...
```

## Migration Path

1. **Phase 1**: Add `contracts/` module with models and loader
2. **Phase 2**: Implement validators for all test types
3. **Phase 3**: Implement asset check generation
4. **Phase 4**: Add decorator pattern (`@with_contract`)
5. **Phase 5**: Integrate with PostgresIOManager (optional enforcement)
6. **Phase 6**: Integrate with OpenLineage emitter (schema facet)
7. **Phase 7**: Documentation and examples

## Design Decisions

### YAML over Python
Contracts are YAML files, not Python code. This keeps them:
- Readable by non-engineers (analysts, stakeholders)
- Diffable in code review
- Separable from implementation logic

### Asset checks over IO manager validation
Validation runs as Dagster asset checks rather than in the IO manager because:
- Better visibility in Dagster UI
- Native alerting integration
- Configurable blocking behavior
- Checks can run independently of materialization

### Severity levels
Two levels (error/warn) rather than more granular options because:
- Maps directly to Dagster's `AssetCheckSeverity`
- Simple mental model: blocks or doesn't block
- Avoids over-engineering

### Contract-per-asset
One contract file per asset rather than a monolithic schema because:
- Co-located with asset code for discoverability
- Independent versioning and ownership
- Easier to review changes

## Data Source Files (`*.source.yaml`)

Data source definitions live in standalone `*.source.yaml` files and are
referenced from pipeline contracts via the `data_source` field. They describe
where data comes from and define historical period boundaries for SCD2
backfills.

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `source_id` | string (UUID) | Stable UUID identifying this data source. Persists across renames so registry history remains correlated. Validated as a UUID at load time. |
| `source_name` | string | Human-readable name for display and logging (e.g., `"cms-asp-crosswalk"`). |
| `periods` | list | Ordered list of period objects (see below). |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `description` | string | Free-text description of the data source. |

### Period Object

Each entry in the `periods` list has these fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | Yes | Path or URL to the source data. |
| `effective_from` | date | Yes | Start date of the period (inclusive). |
| `effective_to` | date | No | End date of the period (exclusive). `null` for the current/open-ended period. At most one period may be open-ended. |
| `partition_key` | string | No | Optional partition key. When set, moncpipelib injects this value as a column during write. |

### Example

```yaml
source_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
source_name: cms-asp-crosswalk
description: CMS ASP crosswalk data source
periods:
  - source: "https://cms.gov/files/asp-2025-h1.zip"
    effective_from: 2025-01-01
    effective_to: 2025-07-01
    partition_key: "2025-H1"
  - source: "https://cms.gov/files/asp-2025-h2.zip"
    effective_from: 2025-07-01
    partition_key: "2025-H2"
```

### Period Registry

When a contract has a `data_source` and the write is called with an
`effective_date` matching a period, moncpipelib auto-registers the period in
the `lineage.period_registry` table. The registry row includes:

- `source_id` -- the UUID from the data source
- `source_name` -- human-readable name from the data source
- `run_id` -- Dagster run ID for audit tracking

## Future Enhancements

These are out of scope for initial implementation but could be added later:

1. **Contract inheritance**: Silver contract inherits from bronze, only specifying changes
2. **Cross-asset validation**: Referential integrity checks between assets
3. **Schema evolution tracking**: Detect and document schema changes over time
4. **Contract generation**: Auto-generate initial contract from DataFrame
5. **dbt integration**: Import/export contracts from dbt schema.yml format

## References

- [Dagster Asset Checks](https://docs.dagster.io/guides/test/asset-checks)
- [dbt Model Contracts](https://docs.getdbt.com/docs/collaborate/govern/model-contracts)
- [Soda Checks Language](https://docs.soda.io/soda-cl/soda-cl-overview.html)
- [Great Expectations](https://docs.greatexpectations.io/)