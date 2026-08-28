# moncpipelib

moncpipelib is a Dagster-based data-pipeline framework: it provides the
resources, IO managers, contract enforcement, lineage, and ingest patterns that
Dagster code locations compose into pipelines. It owns the I/O boundary policy
(streaming-by-default reads/writes against Postgres and blob), the
contract/lineage policy, and the partition/period coordination model.

It is maintained by Model Oncology and developed against our internal
data-pipeline needs, released under the Apache 2.0 license.

## Features

- **PostgreSQL IO Manager** -- write Polars DataFrames to PostgreSQL with multiple write modes (full refresh, upsert, append, partitioned)
- **PostgreSQL Resource** -- streaming/batched reads for large tables via server-side cursors
- **Row-level lineage tracking** -- UUID7-based lineage with foreign key to a centralized lineage table
- **Data contracts** -- declarative YAML-based schema validation, auto-enforced on write
- **OpenLineage integration** -- emit lineage events to Marquez, DataHub, or other backends
- **Data transforms** -- `clean_text`, `safe_decimal`, `safe_bool`, `safe_date`, and more

## Installation

```bash
uv add moncpipelib
# or
pip install moncpipelib
```

Requires Python 3.11+.

To work from source:

```bash
git clone https://github.com/model-oncology-public/moncpipelib
cd moncpipelib
uv sync --all-extras --dev
```

## Quick Start

```python
from dagster import asset, Definitions, EnvVar
import polars as pl
from moncpipelib import PostgresResource, PostgresIOManager, clean_text, safe_decimal

database = PostgresResource(
    host=EnvVar("DB_HOST"), port=EnvVar.int("DB_PORT"),
    user=EnvVar("DB_USER"), password=EnvVar("DB_PASSWORD"),
    database=EnvVar("DB_NAME"),
)

@asset
def orders_bronze(database: PostgresResource) -> pl.DataFrame:
    return database.read_batched_to_dataframe("SELECT * FROM raw.orders")

@asset(io_manager_key="silver_io_manager")
def orders_silver(orders_bronze: pl.DataFrame) -> pl.DataFrame:
    return orders_bronze.select([
        clean_text("order_id"),
        safe_decimal("amount"),
    ])

defs = Definitions(
    assets=[orders_bronze, orders_silver],
    resources={
        "database": database,
        "silver_io_manager": PostgresIOManager(
            postgres_resource=database,
            default_schema="silver",
        ),
    },
)
```

For more examples, see the auto-generated [Cookbook](docs/cookbook.md).

## Documentation

| Topic | Link |
|-------|------|
| Usage examples (auto-generated) | [docs/cookbook.md](docs/cookbook.md) |
| Database resources and IO managers | [docs/best-practices.md](docs/best-practices.md) |
| Data contracts specification | [docs/data-contracts-spec.md](docs/data-contracts-spec.md) |
| Row-level lineage tracking | [docs/lineage-tracking.md](docs/lineage-tracking.md) |
| OpenLineage integration | [docs/openlineage-integration-spec.md](docs/openlineage-integration-spec.md) |
| SCD Type 2 guide | [docs/scd2-guide.md](docs/scd2-guide.md) |
| Security controls | [docs/security.md](docs/security.md) |

## Configuration

moncpipelib uses environment variables for global configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `MONCPIPELIB_DEFAULT_DATABASE` | `analytics` | Default database name for contract sources/sinks that omit one |
| `MONCPIPELIB_OPENLINEAGE_SCHEMA_URL` | (repo URL) | Base URL for OpenLineage custom facet schemas |
| `MONCPIPELIB_OPENLINEAGE_NAMESPACE` | `moncpipelib` | Default namespace for OpenLineage jobs/datasets |
| `MONCPIPELIB_LINEAGE_TABLE` | `data_lineage` | Name of the lineage tracking table |
| `MONCPIPELIB_LINEAGE_SCHEMA` | `lineage` | Schema containing the lineage tracking table |

## Development

```bash
uv sync --all-extras         # Install dev dependencies
uv run pytest                # Run tests
uv run ruff check src tests  # Lint
uv run mypy src              # Type check
uv run ruff format src tests # Format
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License

Apache License 2.0 -- see [LICENSE](LICENSE).
