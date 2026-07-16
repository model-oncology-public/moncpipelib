# UUID Handling in Polars: PostgreSQL, Parquet, and Arrow

This document captures the research and design rationale for how moncpipelib handles
UUID columns across the three primary data surfaces: PostgreSQL reads/writes, Parquet
file serialization, and Arrow/IPC exchange.

## Summary

Polars has no native UUID dtype as of early 2026. The correct representation throughout
this codebase is `pl.String` (36-character hyphenated UUID string, e.g.
`"550e8400-e29b-41d4-a716-446655440000"`). This is not a limitation — it is the
ecosystem-validated approach supported by all downstream consumers.

---

## 1. Dtype Options and Trade-offs

| Dtype | Storage per UUID | Parquet compatible | Postgres write | Notes |
|---|---|---|---|---|
| `pl.String` | 36 bytes + 16-byte view | Yes (UTF8/BYTE_ARRAY) | Yes (implicit cast) | **Recommended** |
| `pl.Binary` | 16 bytes + 16-byte view | Yes (BYTE_ARRAY, no annotation) | Requires bytes conversion | Limited expression support |
| `pl.Object` | ~60-80 bytes (Python heap) | **Panics writer** | Broken | Must never reach a DataFrame |

### Why not `pl.Binary`?

`pl.Binary` stores the raw 16-byte UUID representation and uses 2.25x less memory than
`pl.String`. However:

- Polars' `pl.Binary` is Arrow `LargeBinary` (variable-length), not `FixedSizeBinary(16)`,
  so it does not map to Arrow's canonical `arrow.uuid` extension type anyway.
- The `bin` expression namespace is minimal: `.bin.encode('hex')` produces a 32-char
  lowercase hex string without hyphens, requiring additional expression work to restore
  canonical format.
- Joining and filtering work, but display, contract validation, and all tooling in this
  pipeline assume string UUIDs.
- Parquet writes lose UUID type semantics (stored as plain `BYTE_ARRAY`).

The 20-byte-per-row savings only become meaningful above ~100 million rows. At that
scale, re-evaluate.

### Why `pl.Object` is a hard failure

When `psycopg` returns `uuid.UUID` Python objects without an adapter registered, Polars
stores them as `pl.Object`. This dtype:

- Cannot be cast to `pl.String` (Polars issue #15582, closed "not planned")
- Panics `write_parquet()` with `called Option::unwrap() on a None value` (issue #17486,
  open)
- Cannot be vectorized; every operation falls back to Python iteration
- Cannot be fixed by `schema_overrides` alone — the conversion must happen at the driver
  level before data enters Polars

---

## 2. PostgreSQL Read Path

### The UUID Loader

By default, `psycopg` returns PostgreSQL `UUID` columns as `uuid.UUID` Python objects.
The `PostgresPolarsSchema.register_uuid_adapter()` method registers a per-connection
`psycopg.adapt.Loader` subclass on OID 2950 that decodes the wire bytes to a plain
Python string instead:

```python
class _StringUUIDLoader(psycopg.adapt.Loader):
    def load(self, data):
        return bytes(data).decode("utf-8") if data is not None else None

connection.adapters.register_loader("uuid", _StringUUIDLoader)
```

This loader **must** be registered on every connection before any query whose results
will be loaded into a Polars DataFrame. It is registered automatically in:

- `PostgresResource.get_connection()`
- `PostgresResource.get_connection_raw()`
- `_read_batched_streaming()` via `register_uuid_adapter_sa()`
- `_read_batched_offset()` via `register_uuid_adapter()`

### Schema overrides and `infer_schema_length`

`schema_overrides` pins the OID-to-dtype mapping derived from the cursor description,
ensuring deterministic types regardless of what Polars would infer from sampled data.
For UUID columns, this locks in `pl.String` (OID 2950 → `pl.String` in `OID_MAP`).

`infer_schema_length=0` disables row sampling for schema inference entirely, meaning
Polars relies solely on `schema_overrides` rather than sampling the first N rows. Both
must be used together in `pl.read_database` calls to eliminate any possibility of a
`pl.Object` UUID column.

### ConnectorX alternative

When using `pl.read_database_uri()` with a ConnectorX connection string, UUID-to-String
conversion happens at the Rust transport layer automatically. No adapter is needed. This
is more reliable for ad-hoc reads but is not the primary read path in this library.

---

## 3. PostgreSQL Write Path

### The `::VARCHAR` cast bug

Polars' `write_database()` via SQLAlchemy has an unfixed bug (issue #21438, closed "not
planned") where it emits `%(col)s::VARCHAR` parameter casts that fail when the target
column is typed `UUID` in PostgreSQL. This is a fundamental incompatibility with the
SQLAlchemy write path.

### How this codebase avoids it

The `PostgresIOManager` uses `psycopg`'s `cursor.executemany` (or `cursor.copy()` for
the COPY path) directly for all writes, bypassing `write_database` entirely. PostgreSQL
performs an implicit cast from `VARCHAR` to `UUID` when receiving string values, making
`pl.String` UUIDs safe to write to native `UUID` columns.

### ADBC (future consideration)

If ADBC is ever adopted for higher-throughput bulk writes (it uses PostgreSQL `COPY`
internally), UUID columns require special handling:

- ADBC has no UUID type mapping; it does not recognize OID 2950
- Writing `pl.String` UUIDs via ADBC triggers the same `::VARCHAR` cast failure
- The correct approach for ADBC writes is to convert UUID columns to raw bytes before
  ingestion, or define target DDL columns as `TEXT` rather than `UUID`

---

## 4. Parquet Serialization

### Parquet UUID logical type specification

The Parquet format defines UUID as `FIXED_LEN_BYTE_ARRAY(16)` with a UUID logical type
annotation (big-endian byte order). This is the compact, type-annotated representation.

### What Polars actually writes

Polars can **read** the Parquet UUID logical type but **cannot write** it. The behavior
by dtype:

| Input dtype | Parquet physical type | Logical type | Readable by all consumers |
|---|---|---|---|
| `pl.String` | `BYTE_ARRAY` | `UTF8` / `STRING` | Yes |
| `pl.Binary` | `BYTE_ARRAY` | None | Yes (as bytes) |
| `pl.Object` | — | — | **Panics the writer** |

For any asset that writes Parquet files containing UUID columns, use `pl.String`. The
UTF8-annotated `BYTE_ARRAY` is lossless and universally readable by DuckDB, Spark,
PyArrow, and all other Parquet consumers.

### PyArrow 21.0 note

PyArrow fixed a parallel issue (Arrow #46469) in version 21.0 (May 2025): `pa.uuid()`
extension arrays now correctly write the Parquet UUID logical type annotation. If Polars
adopts the updated Arrow serialization path, the compact UUID Parquet type may eventually
become writable. Until then, `pl.String` is the only safe choice.

---

## 5. Arrow/IPC Exchange

Arrow's canonical UUID extension type (`arrow.uuid`) uses `FixedSizeBinary(16)` storage
with big-endian byte order. Polars added partial Arrow extension type support (PR #25322)
but does not expose a Python-facing `pl.Uuid` dtype. When Polars encounters
`FixedSizeBinary(16)` from an Arrow source, it stores it as `pl.Binary` (variable-length
`LargeBinary`), losing the fixed-width guarantee and the `arrow.uuid` extension metadata.

For IPC exchange with systems that use Arrow natively (e.g. Dagster's Arrow-backed IO,
ConnectorX zero-copy path), `pl.String` UUID columns survive the round-trip correctly.
`pl.Binary` also survives but loses the `arrow.uuid` type annotation.

---

## 6. Performance Characteristics

With Polars' string view architecture (introduced 2024), `pl.String` UUID performance is
strong:

- Strings longer than 12 characters (UUIDs are 36) use a 16-byte view pointer into a
  secondary buffer — a single level of indirection.
- Filter and gather operations are O(n) in row count, not O(n * string_length) — a
  critical improvement for UUID-heavy pipelines.
- The memory difference between `pl.String` (36 bytes) and `pl.Binary` (16 bytes) is
  20 bytes per UUID. At 10 million rows this is ~190 MB; meaningful only at very large
  scale.

---

## 7. Known Open Issues (upstream Polars)

| Issue | Status | Impact |
|---|---|---|
| `pl.Object` (uuid.UUID) cannot cast to `pl.String` — #15582 | Closed "not planned" | Mitigated by UUID adapter on every connection |
| `write_parquet` panics on `pl.Object` UUID columns — #17486 | Open | Mitigated by ensuring UUIDs never reach `pl.Object` |
| `write_database` emits `::VARCHAR` that fails on UUID columns — #21438 | Closed "not planned" | Mitigated by using `execute_values` directly |
| No native `pl.Uuid` dtype — #7175, #9112 | Open wish | Mitigated by `pl.String` + `schema_overrides` |
| Polars cannot write Parquet UUID logical type | Not tracked upstream | Mitigated by writing `pl.String` |

---

## 8. Checklist for New Code

When writing new Polars code that touches UUID columns:

- [ ] Are you reading from `psycopg`? Call `register_uuid_adapter()` before the query.
- [ ] Are you using `pl.read_database`? Pass both `schema_overrides` and
  `infer_schema_length=0`.
- [ ] Are you writing to Parquet? Ensure UUID columns are `pl.String` before
  `write_parquet()`.
- [ ] Are you writing to PostgreSQL via `execute_values`? `pl.String` UUIDs work. If
  using ADBC, convert to bytes first.
- [ ] Did a UUID column come back as `pl.Object`? This is a bug — fix it at the driver
  level, not with `map_elements`.