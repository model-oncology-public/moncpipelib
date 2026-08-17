# Security Controls

This document describes the security controls implemented in moncpipelib to
support **HIPAA** compliance (statutory); the organization is pursuing a
**SOC 2 Type 2** attestation and **HITRUST i1** certification.

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

## Write-path Audit Volume

PostgreSQL session auditing (e.g. `pgaudit`) logs one entry per statement
*execution*. Before this control, the bulk-insert paths could turn a single
logical load into one statement execution per row: a 392k-row streamed load
produced roughly 731k audit-log lines, a volume that became an operational
and storage problem in its own right, independent of the load's own cost.

Two mechanisms bound the number of statement executions a bulk insert
produces:

- **COPY sizing accounts for the whole stream.** Bulk inserts choose between
  the `COPY` protocol (one statement regardless of row count) and a
  parameterized multi-row `INSERT` path based on row count. For a streamed,
  batched write, the caller may supply an estimated stream total alongside
  each batch's own row count; the larger of the two sizes the decision, so a
  large load broken into many small batches still reaches `COPY` instead of
  being sized batch-by-batch forever.
- **The parameterized INSERT path pages multiple rows per statement.**
  Instead of one statement execution per row, rows are packed into
  multi-row `INSERT ... VALUES (...), (...), ...` executions, configurable
  via a rows-per-page setting. Each page is one statement execution and
  therefore one audit-log entry, so audit volume divides by (approximately)
  the page size. The page size is automatically capped so no single
  statement exceeds PostgreSQL's per-statement bind-parameter ceiling.

Choosing between the two insert mechanisms never changes what is written:
`COPY` and the parameterized path serialize SQL NULL, empty string, and
literal text identically (both use the encoding in
`io_managers/writers.py::serialize_for_staging_copy`), so which one a given
batch takes changes only statement shape and audit volume, never the row
values landed.

In both cases, row values are always passed as bind parameters -- never
formatted into the SQL text -- at every page size and for every insert
method, so the audited *statement text* never contains row data. Whether
row *values* reach the audit log at all depends on a separate setting,
`pgaudit.log_parameter`: with it off (the operational posture this project
runs under), bind values are absent from the audit log entirely. If
parameter logging were enabled instead, bind values WOULD appear in audit
entries, and paging concentrates up to a full page of values into each
entry -- so the posture that keeps row values out of the audit log
altogether is `pgaudit.log_parameter` off, not the statement-text guarantee
by itself. See the `_executemany_bulk` docstring for the same scoping.

## Dependency Security

All dependencies are pinned via `uv.lock` and audited regularly. The project
uses `uv` for reproducible builds and dependency resolution.
