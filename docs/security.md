# Security Controls

This document describes the security controls implemented in moncpipelib to
support **HIPAA**, **SOC 2**, **ISO 27001**, and **HITRUST** compliance.

## PII Column Tracking

moncpipelib implements a declarative PII inventory through data contracts. This
serves as a compliance control ensuring that personally identifiable information
(PII) and protected health information (PHI) are identified, tracked, and
protected at every layer of the data platform.

### Safe-by-Default Design

Columns default to `pii: true`. If a column is not explicitly annotated with
`pii: false` in its data contract, it is treated as PII. This means:

- Forgetting to annotate a column results in masking and warnings, not exposure.
- Engineers must explicitly opt columns **out** of PII protection.
- Unannotated columns trigger a warning at every contract load.

### Control Layers

PII tracking is enforced across multiple layers:

| Layer | Control | Description |
|-------|---------|-------------|
| **Data Contracts** | `pii` field on `Column` | Authoritative source of PII classification. YAML-based, diffable in code review. |
| **PostgreSQL Catalog** | `COMMENT ON COLUMN` | PII tags (`PII:true`/`PII:false`) synced to column comments at write time. Enables catalog-level auditing. |
| **OpenLineage** | `ColumnClassificationFacet` | PII column list emitted with every lineage event. Visible in Marquez/DataHub. |
| **Dagster Metadata** | `pii_columns`, `pii_column_count` | PII metadata attached to every materialization. Visible in Dagster UI. |
| **Log Rendering** | `polars_to_md()` | DataFrame-to-markdown utility that masks PII columns by default when a contract is provided. |
| **Drift Detection** | `_check_pii_drift()` | Warns at `load_input()` when upstream PII columns flow into downstream assets without PII tracking. |

### Contract PII Annotations

In a data contract YAML file, annotate each column with `pii: true` or
`pii: false`:

```yaml
schema:
  columns:
    - name: patient_id
      type: string
      nullable: false
      pii: true   # PHI -- will be masked in logs, tracked in catalog

    - name: claim_id
      type: string
      nullable: false
      pii: false   # Business identifier, not PII
```

Columns that omit `pii` default to `true` and trigger a warning:

```
WARNING Contract 'claims_bronze': columns ['provider_name'] have no explicit
'pii' annotation. They default to pii=true (treated as PII). Add 'pii: false'
to opt out or 'pii: true' to suppress this warning.
```

### PostgreSQL Catalog Sync

After every successful data write, moncpipelib syncs PII annotations to
PostgreSQL column comments using `COMMENT ON COLUMN`. Comments use a
`KEY:value` tag format (semicolon-delimited) so other metadata can coexist:

```
PII:true;OWNER:data-eng
```

This sync is metadata-only -- failures are logged as warnings but do **not**
roll back the data write.

### PII Drift Detection

When a downstream asset loads data from an upstream asset, moncpipelib compares
PII annotations between their contracts. If an upstream PII column exists in
the downstream schema but is not marked as PII there, a warning is logged:

```
WARNING PII drift detected: columns ['patient_id'] are PII in upstream
'claims_bronze' but NOT marked as PII in downstream 'claims_silver'.
Review downstream contract PII annotations.
```

This check runs at `load_input()` time when `enforce_contracts` is `WARN` or
`ERROR` (not `SILENT`).

### Log-Safe Rendering

The `polars_to_md()` function renders Polars DataFrames as markdown tables with
automatic PII masking. When a data contract is provided, columns with
`pii: true` are replaced with `***`:

```python
from moncpipelib.rendering import polars_to_md

md = polars_to_md(df, contract=contract)
# patient_id and name columns show *** instead of real values
```

This prevents accidental PII/PHI exposure in Dagster logs, notebooks, and
monitoring dashboards.

## Data Lineage

Row-level lineage tracking via UUID7 provides auditability for every row written
through moncpipelib. See the lineage module documentation for details.

## Dependency Security

All dependencies are pinned via `uv.lock` and audited regularly. The project
uses `uv` for reproducible builds and dependency resolution.
