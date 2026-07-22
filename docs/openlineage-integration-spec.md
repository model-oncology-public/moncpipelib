# OpenLineage Integration Specification

This document specifies how moncpipelib emits OpenLineage events, enabling cross-system lineage visibility while maintaining the existing row-level tracking capabilities.

> **Reconciled 2026-07-03 (issue #393):** this document originally predated the
> implementation and had drifted from the shipped API. It now describes the
> implemented surface as of v0.40.x. The tested cookbook examples
> (`docs/cookbook.md`, generated from `tests/cookbook/test_openlineage_cookbook.py`)
> are ground truth; if this document and the cookbook disagree, trust the cookbook.

## Overview

### Goals

1. Emit OpenLineage-compliant events when assets are materialized
2. Map moncpipelib lineage metadata to standard and custom OpenLineage facets
3. Support multiple backends (Marquez, DataHub, custom HTTP endpoints)
4. Emission is automatic once `openlineage_url` is configured on the resource; leave it unset to disable
5. Keep dependencies minimal and optional

### Non-Goals

- Replace the existing row-level lineage tracking (complementary, not replacement)
- Implement a full OpenLineage backend/server

## Backend Requirements

**You must deploy an OpenLineage-compatible backend to receive events.** Options:

### Marquez (Recommended for simplicity)

Lightweight, purpose-built for OpenLineage. Deploy via Helm:

```bash
helm repo add marquez https://marquezproject.github.io/marquez
helm install marquez marquez/marquez
```

Components: API server + PostgreSQL. Provides a web UI for lineage visualization.

### DataHub (If you need a full data catalog)

Heavier deployment but includes data catalog, governance features. Has native OpenLineage endpoint at `/openlineage`.

### Why a separate backend?

Dagster's event log captures run metadata but does not:
- Store events in OpenLineage format
- Provide cross-system lineage (only sees Dagster assets)
- Emit to external systems

The `openlineage-dagster` package that existed used a sensor to tail Dagster's event log and convert events, but it was removed from the OpenLineage project. Emitting directly from the write path (as implemented) is more reliable and immediate.

## Architecture

Emission lives on `PostgresResource` write paths (`write()` /
`_write_single()` / `_write_batched()`), not on the IO manager -- the IO
manager delegates all database work, including lineage and OpenLineage, to
the resource:

```
PostgresResource write path
         │
         ├──► emitter.emit_start(job_name, run_id)          # before the write
         │
         ├──► LineageTracker.create_lineage_record()
         │    └── Insert to lineage.data_lineage (existing)
         │
         ├──► emitter.emit_complete(job_name, run_id, ...)  # after commit
         │
         └──► emitter.emit_fail(job_name, run_id, ...)      # on write error
```

## Dependencies

`openlineage-python` is an optional dependency. If it is not installed, the
resource silently skips emitter creation (`ImportError` is swallowed) and
writes proceed without OpenLineage events.

## Module Structure

```
src/moncpipelib/
├── lineage/
│   ├── __init__.py          # Exports OpenLineageEmitter, OpenLineageConfig, facets
│   ├── tracker.py           # Row-level lineage (no OpenLineage coupling)
│   └── openlineage.py       # Emitter, config, custom facets
```

## API Specification

### OpenLineageConfig (Dagster Resource)

```python
from dagster import ConfigurableResource

class OpenLineageConfig(ConfigurableResource):
    """Configuration for OpenLineage event emission."""

    url: str
    """Base URL of the OpenLineage backend (e.g., "http://marquez:5000").
    The OpenLineage client appends the API path itself -- do NOT include
    /api/v1/lineage."""

    namespace: str = ...  # default from MONCPIPELIB_OPENLINEAGE_NAMESPACE env var, else "moncpipelib"
    """Namespace for jobs and datasets. Used verbatim -- no layer suffixing."""

    api_key: str | None = None
    """Optional API key for authenticated endpoints."""

    timeout: float = 10.0
    """HTTP request timeout in seconds."""

    enabled: bool = True
    """Set False to disable emission without removing config."""
```

There is no `producer` field and no `get_emitter()` method; construct
`OpenLineageEmitter(config)` directly. The producer URI on emitted events is
derived from the schema URL base (see Custom Facets below).

### OpenLineageEmitter Class

Location: `src/moncpipelib/lineage/openlineage.py`

The emitter exposes three methods that share a `run_id`: `emit_start()`
returns (or accepts) the run ID, and `emit_complete()` / `emit_fail()` take
it back. There is no Dagster `context` parameter anywhere on the surface.

```python
class OpenLineageEmitter:
    """Emits OpenLineage events to external lineage backends."""

    def __init__(self, config: OpenLineageConfig) -> None: ...

    def emit_start(
        self,
        job_name: str,
        run_id: str | None = None,
        input_datasets: list[str] | None = None,
    ) -> str:
        """Emit a START run event. Returns the run_id (generated if not
        provided) for the paired emit_complete/emit_fail call."""

    def emit_complete(
        self,
        job_name: str,
        run_id: str,
        output_dataset: str | None = None,   # e.g., "bronze.orders"
        row_count: int | None = None,
        df: pl.DataFrame | None = None,      # for schema facet extraction
        lineage_id: str | None = None,
        lineage_key: str | None = None,
        layer: str | None = None,
        is_backfill: bool = False,
        pipeline_id: str | None = None,
        parent_lineage_ids: list[str] | None = None,
        source_file: str | None = None,
        data_date: str | None = None,
        input_datasets: list[str] | None = None,
        pii_columns: list[str] | None = None,
    ) -> None:
        """Emit a COMPLETE run event with output dataset facets.
        (#391 adds a phi_columns parameter alongside pii_columns.)"""

    def emit_fail(
        self,
        job_name: str,
        run_id: str,
        error_message: str | None = None,
        input_datasets: list[str] | None = None,
    ) -> None:
        """Emit a FAIL run event. Called by the resource write paths when a
        write raises; not currently demonstrated in the cookbook."""
```

Usage (matches the tested cookbook example):

```python
from moncpipelib.lineage import OpenLineageEmitter, OpenLineageConfig

config = OpenLineageConfig(
    url="http://marquez:5000",  # base URL only
    namespace="analytics",
)
emitter = OpenLineageEmitter(config)

run_id = emitter.emit_start(job_name="orders_bronze")
emitter.emit_complete(
    job_name="orders_bronze",
    run_id=run_id,
    output_dataset="bronze.orders",
    row_count=1000,
)
```

## Custom Facets

Custom facet schema URLs default to the GitHub tree URL base:

`https://github.com/model-oncology-public/moncpipelib/tree/main/schemas/openlineage/`

The base is overridable via the `MONCPIPELIB_OPENLINEAGE_SCHEMA_URL`
environment variable (see `config.py`). Each facet serializes
`_schemaURL` as `{base}{FacetName}/1-0-0/{FacetName}.json`, and the event
`producer` is `{base}producer`.

> Note: the JSON schema files live flat on disk at
> `schemas/openlineage/1-0-0/{FacetName}.json`, while the serialized
> `_schemaURL` uses a `{FacetName}/1-0-0/` prefix. The schemas' `$id` values
> use the flat layout. This mismatch is cosmetic (nothing dereferences the
> URLs at runtime) but is worth knowing when looking up a schema by URL.

### MoncpipelibLineageFacet

Custom facet for row-level lineage metadata not covered by standard facets.
Fields as implemented and serialized by `to_dict()`:

```python
@dataclass
class MoncpipelibLineageFacet:
    lineage_id: str                   # UUID7 from moncpipelib
    lineage_key: str                  # Composite backup key
    layer: str                        # bronze/silver/gold
    is_backfill: bool = False
    pipeline_id: str | None = None    # emitted only when set
    parent_lineage_ids: list[str] = field(default_factory=list)
    # _schemaURL is fixed, init=False
```

The JSON schema (`schemas/openlineage/1-0-0/MoncpipelibLineageFacet.json`)
additionally allows optional legacy properties (`backfill_reason`,
`replaces_lineage_id`, `transformation_type`) that the Python facet does not
currently emit.

### DataPartitionFacet

Custom facet for data date/range information. `to_dict()` emits only the
keys that are set:

```python
@dataclass
class DataPartitionFacet:
    data_date: str | None = None         # ISO format date
    data_date_start: str | None = None   # ISO format date (range start)
    data_date_end: str | None = None     # ISO format date (range end)
```

The JSON schema allows an optional `partition_column` property that the
Python facet does not currently emit.

### SourceFileFacet

Custom facet for source file tracking. `to_dict()` emits only set keys:

```python
@dataclass
class SourceFileFacet:
    source_file: str | None = None    # File path or name
    source_system: str | None = None  # sftp, api, blob, etc.
    file_format: str | None = None    # csv, parquet, json, etc.
```

The JSON schema allows an optional `file_size_bytes` property that the
Python facet does not currently emit.

### ColumnClassificationFacet

Custom facet recording which output columns are classified as PII, populated
from the data contract's per-column `pii` flags at write time
(`get_pii_column_names()`):

```python
@dataclass
class ColumnClassificationFacet:
    pii_columns: list[str] = field(default_factory=list)
    # (#391 adds phi_columns: list[str] alongside pii_columns)
```

## Facet Mapping

Map moncpipelib fields to OpenLineage standard and custom facets:

| moncpipelib field | OpenLineage facet | Notes |
|-------------------|-------------------|-------|
| `asset_name` | Job name | Job namespace is the configured namespace, used verbatim |
| `layer` | MoncpipelibLineageFacet.layer | Custom facet field (NOT a namespace suffix) |
| `run_id` | RunEvent.run.runId | Direct mapping |
| `row_count` | OutputStatisticsOutputDatasetFacet.rowCount | Standard facet |
| `source_file` | SourceFileFacet (custom) | Custom facet |
| `data_date` | DataPartitionFacet.data_date | Custom facet |
| `is_backfill` | MoncpipelibLineageFacet.is_backfill | Custom facet |
| `pipeline_id` | MoncpipelibLineageFacet.pipeline_id | Custom facet, emitted when set |
| `parent_lineage_ids` | MoncpipelibLineageFacet.parent_lineage_ids | Custom facet |
| `lineage_id` | MoncpipelibLineageFacet.lineage_id | Custom facet |
| `lineage_key` | MoncpipelibLineageFacet.lineage_key | Custom facet |
| contract `pii` flags | ColumnClassificationFacet.pii_columns | Custom facet |
| DataFrame schema | SchemaDatasetFacet | Standard facet, from the `df` param |

## Input and Output Datasets

`emit_start()` and `emit_complete()` accept `input_datasets` as a list of
dataset **names** (e.g., `["bronze.claims"]`); each becomes an OpenLineage
input dataset in the configured namespace. Output datasets are named by the
`output_dataset` string (e.g., `"bronze.orders"`) -- the schema/table
qualification lives in the dataset *name*, not the namespace.

## Integration with PostgresResource

OpenLineage is configured with three fields on `PostgresResource` (the IO
manager has no OpenLineage fields -- it delegates writes to the resource):

```python
from dagster import Definitions, EnvVar
from moncpipelib import PostgresIOManager, PostgresResource

database = PostgresResource(
    host=EnvVar("DB_HOST"),
    port=5432,
    user=EnvVar("DB_USER"),
    password=EnvVar("DB_PASSWORD"),
    database=EnvVar("DB_NAME"),
    openlineage_url=EnvVar("OPENLINEAGE_URL"),   # e.g., http://marquez:5000
    openlineage_namespace="model-oncology",
    # openlineage_api_key=EnvVar("OPENLINEAGE_API_KEY"),  # if required
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

Behavior:

- If `openlineage_url` is unset (default `None`), no emitter is created and
  no events are emitted.
- If `openlineage-python` is not installed, emitter creation is skipped
  silently.
- The write path emits START before the write and COMPLETE after commit,
  threading `lineage_id` / `lineage_key` / `layer` / `pii_columns` and other
  metadata into the COMPLETE event's output facets; on error it emits FAIL.
- The layer comes from the target schema when it is a recognized layer name,
  falling back to the contract's `layer:` field -- it is not configured on
  the emitter.

### Disabled (Local Development)

Simply leave `openlineage_url` unset on the resource. To keep a URL
configured but pause emission, construct an emitter whose config has
`enabled=False` -- `emit_*` become no-ops.

### Environment Variables

For Kubernetes/production, configure via environment:

```yaml
# values.yaml or ConfigMap
env:
  OPENLINEAGE_URL: "http://marquez.lineage.svc.cluster.local:5000"
  MONCPIPELIB_OPENLINEAGE_NAMESPACE: "model-oncology"
```

## Error Handling

OpenLineage emission is **non-blocking** and **fail-safe**:

1. All emission is wrapped in try/except inside `_emit_event`
2. Failures log warnings, never raise
3. HTTP timeouts (default 10s) prevent hanging
4. No retry queue -- if the backend is down, events are lost (acceptable for observability data)

## Testing

Unit tests live in `tests/test_openlineage.py` (facet serialization, emitter
enable/disable behavior, custom facet construction, graceful error handling).
Tested end-to-end examples live in `tests/cookbook/test_openlineage_cookbook.py`
and render into `docs/cookbook.md`.

## JSON Schema Files

Schemas for the custom facets are version-controlled at
`schemas/openlineage/1-0-0/`:

- `MoncpipelibLineageFacet.json`
- `DataPartitionFacet.json`
- `SourceFileFacet.json`
- `ColumnClassificationFacet.json`

Each schema's `$id` points at the GitHub tree URL
(`https://github.com/model-oncology-public/moncpipelib/tree/main/schemas/openlineage/1-0-0/...`).
The schema files are the source of truth for facet shape; update them
whenever the Python facet classes in `src/moncpipelib/lineage/openlineage.py`
change (see CLAUDE.md "Update OpenLineage facet schemas").

## Design Decisions

### Opt-in by URL
Emission requires `openlineage_url` on the resource. An unset URL (the
default) disables OpenLineage entirely -- there is no separate global toggle
to forget about.

### Resource-based emission
Events are emitted from `PostgresResource` write paths so that every write
route (IO manager delegation, direct `database.write(...)`, batched writes)
emits identically. The IO manager carries no OpenLineage configuration.

### Fire-and-forget emission
Failed OpenLineage events log warnings but don't fail the asset. Observability
shouldn't break pipelines. No retry queue -- if Marquez is down, events are
lost (acceptable for observability data).

### Namespace used verbatim
The configured namespace is applied to jobs and datasets as-is. Layer lives
in `MoncpipelibLineageFacet.layer` and in dataset names (`bronze.orders`),
not in namespace suffixes.

## References

- [OpenLineage Spec](https://openlineage.io/docs/spec/object-model)
- [Custom Facets](https://openlineage.io/docs/spec/facets/custom-facets/)
- [openlineage-python](https://pypi.org/project/openlineage-python/)
- [Marquez](https://marquezproject.ai/)
- [DataHub OpenLineage Integration](https://docs.datahub.com/docs/lineage/openlineage)
