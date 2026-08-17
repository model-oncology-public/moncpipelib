"""Cookbook entry for :func:`read_latest_partition`.

Code between ``# --- cookbook:start ---`` / ``# --- cookbook:end ---``
is extracted into ``docs/cookbook.md`` by the cookbook plugin.

This recipe shows the "non-partitioned reference silver" pattern: the
silver answers "what does code X mean today?" -- there is no SCD2,
there are no per-partition silvers, the table is truncated and replaced
on every materialization from the latest bronze partition.

The example wires :func:`read_latest_partition` -> :func:`transform_batched`
-> ``database.write(target=..., full_refresh=True)``.  Real silvers
inject a :class:`PostgresResource`; this test uses a tiny stand-in that
implements the same surface so the cookbook example runs in CI without
a database.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.mark.cookbook(
    title="Non-partitioned reference silver reading the latest bronze partition",
    description=(
        "Some bronze tables are append-partitioned reference "
        "dictionaries (ICD-O-3 morphology / topography, NDC lookups, "
        "etc.) where the silver consumer never wants history -- it "
        "just wants the freshest snapshot.  "
        "``read_latest_partition`` streams every row of the upstream "
        "bronze's latest partition (``WHERE load_period = (SELECT "
        "MAX(load_period) FROM ...)``), pairs cleanly with "
        "``transform_batched`` for per-batch cleaning, and feeds a "
        "``full_refresh=True`` write so the silver is truncated and "
        "replaced atomically.  An empty bronze raises "
        "``EmptyPartitionedTableError`` (a ``LookupError``) so the "
        "asset fails fast with a useful pointer instead of silently "
        "wiping the silver."
    ),
    category="reference",
)
def test_cookbook_non_partitioned_reference_silver() -> None:
    # --- cookbook:start ---
    import polars as pl

    from moncpipelib import (
        BatchedDataFrame,
        EmptyPartitionedTableError,
        read_latest_partition,
        transform_batched,
    )

    # --- 1. The silver asset --------------------------------------------------
    # In production this is a Dagster @asset with
    #   io_manager_key="postgres_io_manager", required_resource_keys={"database"}
    # The asset body is the four lines after the docstring; everything else
    # in this cookbook is plumbing so the example runs in CI.
    def seer_icdo3_morphology_silver(
        context: Any,  # AssetExecutionContext in production
        database: Any,  # PostgresResource in production
    ) -> BatchedDataFrame:
        """Current-snapshot ICD-O-3 morphology lookup."""
        batches = read_latest_partition(
            database,
            source_table="reference_bronze.icdo3_morphology",
            partition_column="load_period",  # default; here for explicitness
            columns=("code", "label", "category"),
            batch_size=50_000,
            context=context,
        )
        return transform_batched(batches, transform_fn=_clean_morphology_batch)

    def _clean_morphology_batch(df: pl.DataFrame) -> pl.DataFrame:
        """Cleanup applied independently to each streamed batch.

        Cross-batch aggregations (global dedup, percentiles, etc.) are
        not supported here -- if you need them, materialise the full
        partition with ``read_batched_to_dataframe`` instead.
        """
        return (
            df.lazy()
            .with_columns(
                pl.col("code").str.strip_chars(),
                pl.col("label").str.strip_chars(),
            )
            .collect(engine="streaming")
        )

    # --- 2. Tiny stand-in resource so the example runs in CI ------------------
    # The real silver injects a PostgresResource.  This stand-in matches the
    # surface ``read_latest_partition`` actually touches: ``get_connection()``
    # for the precheck, and the resource passed to ``read_batched`` for the
    # streaming SELECT.  The "database" here is a pure Python dict of
    # partition -> rows.
    bronze_rows = {
        "2026-03-01": [
            ("8000/0", "Neoplasm, benign (old taxonomy)", "morphology"),
        ],
        "2026-05-01": [
            ("8000/0", "Neoplasm, benign", "morphology"),
            ("8140/3", "Adenocarcinoma, NOS", "morphology"),
            ("9590/3", "Malignant lymphoma, NOS", "morphology"),
        ],
    }

    class _StandInResource:
        """Minimal stand-in -- mirrors the surface the helper uses."""

        def __init__(self, partitions: dict[str, list[tuple[str, str, str]]]) -> None:
            self._partitions = partitions

        def get_connection(self) -> Any:
            # Used by the precheck.  Returns a context-manager that yields
            # a connection with a single-shot cursor returning MAX(load_period).
            partitions = self._partitions
            latest = max(partitions) if partitions else None

            class _Cursor:
                def __enter__(self) -> _Cursor:
                    return self

                def __exit__(self, *args: Any) -> None:
                    return None

                def execute(self, _query: Any) -> None:
                    return None

                def fetchone(self) -> tuple[Any]:
                    return (latest,)

            class _Conn:
                def __enter__(self) -> _Conn:
                    return self

                def __exit__(self, *args: Any) -> None:
                    return None

                def cursor(self) -> _Cursor:
                    return _Cursor()

            return _Conn()

    # ``read_latest_partition`` then forwards the resource to ``read_batched``
    # for the streaming SELECT.  In this cookbook we patch that out -- a real
    # asset never needs to.
    from unittest.mock import patch

    def _fake_read_batched(
        _query: str,
        _connection: Any,
        *,
        batch_size: int,
        context: Any = None,  # noqa: ARG001
    ) -> Any:
        del batch_size
        latest_rows = bronze_rows[max(bronze_rows)]
        # Single batch is fine; for larger partitions, ``read_batched``
        # yields multiple frames sized to ``batch_size``.
        yield pl.DataFrame(
            latest_rows,
            schema=["code", "label", "category"],
            orient="row",
        )

    # --- 3. Run the asset -----------------------------------------------------
    database = _StandInResource(bronze_rows)
    # ``read_latest_partition`` returns an iterator; the SELECT runs lazily on
    # the first ``next(...)``, so the patch needs to stay active until the
    # iterator is consumed.
    with patch("moncpipelib.reference.read_batched", _fake_read_batched):
        batched = seer_icdo3_morphology_silver(context=None, database=database)
        # ``BatchedDataFrame`` carries the streaming iterator that the IO
        # manager consumes during ``database.write(..., full_refresh=True)``.
        # Here we collect for assertion.
        result = pl.concat(list(batched.batches))

    # Only the latest partition's rows -- the old "2026-03-01" entry is
    # excluded by the ``WHERE load_period = (SELECT MAX(load_period) ...)``.
    assert result.shape == (3, 3)
    assert result.columns == ["code", "label", "category"]
    assert set(result["code"].to_list()) == {"8000/0", "8140/3", "9590/3"}

    # --- 4. Empty bronze fails fast ------------------------------------------
    # If the bronze has no rows yet, the precheck raises
    # ``EmptyPartitionedTableError`` (a ``LookupError`` subclass) naming the
    # source table and the partition column -- the silver asset fails with a
    # useful pointer instead of silently truncating to zero rows.
    empty_database = _StandInResource(partitions={})
    with pytest.raises(EmptyPartitionedTableError) as excinfo:
        list(
            read_latest_partition(
                empty_database,
                source_table="reference_bronze.icdo3_morphology",
            )
        )
    assert "reference_bronze.icdo3_morphology" in str(excinfo.value)
    assert "load_period" in str(excinfo.value)

    # --- 5. Wire it up in Dagster --------------------------------------------
    # The full asset is one decorator away.  In production:
    #
    #   @asset(
    #       io_manager_key="postgres_io_manager",
    #       required_resource_keys={"database"},
    #       metadata={"write_mode": "full_refresh", "schema": "reference_silver"},
    #   )
    #   def seer_icdo3_morphology(context, database) -> BatchedDataFrame:
    #       ...the body above...
    #
    # The IO manager's ``full_refresh`` mode truncates the silver and
    # bulk-loads the streamed batches in a single transaction -- consumers
    # see either the prior snapshot or the new one, never a half-written
    # state.
    # --- cookbook:end ---
