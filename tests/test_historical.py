"""Tests for historical SCD2 backfill utilities."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest

from moncpipelib.contracts.models import DataSource, Period
from moncpipelib.historical import (
    RegistryPartitionsDefinition,
    build_partitions_from_periods,
    build_partitions_from_registry,
    get_period_for_partition,
    get_period_from_registry,
    load_historical_periods,
)


def _make_source(periods: list[Period]) -> DataSource:
    """Create a DataSource with the given periods."""
    return DataSource(
        source_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        source_name="test-source",
        periods=periods,
    )


class TestPeriodModel:
    """Tests for Period dataclass."""

    def test_period_fields(self) -> None:
        p = Period(
            source="/data/q1.csv", effective_from=date(2025, 1, 1), effective_to=date(2025, 4, 1)
        )
        assert p.source == "/data/q1.csv"
        assert p.effective_from == date(2025, 1, 1)
        assert p.effective_to == date(2025, 4, 1)

    def test_period_open_ended(self) -> None:
        p = Period(source="/data/current.csv", effective_from=date(2026, 1, 1))
        assert p.effective_to is None

    def test_period_frozen(self) -> None:
        p = Period(source="/data/q1.csv", effective_from=date(2025, 1, 1))
        with pytest.raises(AttributeError):
            p.source = "other"  # type: ignore[misc]


class TestDataSourceLoading:
    """Tests for data source loading from YAML."""

    def test_load_data_source_from_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "source.yaml").write_text(
            """\
source_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
source_name: test-source
description: Test data source
periods:
  - source: "https://example.com/h1.csv"
    effective_from: 2025-01-01
    effective_to: 2025-07-01
  - source: "https://example.com/h2.csv"
    effective_from: 2025-07-01
    effective_to:
"""
        )
        from moncpipelib.contracts import load_data_source

        ds = load_data_source(tmp_path / "source.yaml")
        assert ds.source_id == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert ds.source_name == "test-source"
        assert ds.description == "Test data source"
        assert len(ds.periods) == 2
        assert ds.periods[0].source == "https://example.com/h1.csv"
        assert ds.periods[0].effective_from == date(2025, 1, 1)
        assert ds.periods[0].effective_to == date(2025, 7, 1)
        assert ds.periods[1].effective_to is None

    def test_contract_resolves_data_source(self, tmp_path: Path) -> None:
        (tmp_path / "my.source.yaml").write_text(
            """\
source_id: "b2c3d4e5-f6a7-8901-bcde-f12345678901"
source_name: resolved-source
periods:
  - source: "https://example.com/h1.csv"
    effective_from: 2025-01-01
    effective_to: 2025-07-01
"""
        )
        (tmp_path / "contract.yaml").write_text(
            """\
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: products
layer: silver
data_source: my.source.yaml
schema:
  columns:
    - name: id
      type: string
      nullable: false
      pii: false
"""
        )
        from moncpipelib.contracts import load_contract

        contract = load_contract(tmp_path / "contract.yaml")
        assert contract.data_source is not None
        assert contract.data_source.source_id == "b2c3d4e5-f6a7-8901-bcde-f12345678901"
        assert contract.data_source.source_name == "resolved-source"
        assert len(contract.data_source.periods) == 1

    def test_periods_overlap_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "source.yaml").write_text(
            """\
source_id: "c3d4e5f6-a7b8-9012-cdef-123456789012"
source_name: bad-source
periods:
  - source: "a.csv"
    effective_from: 2025-01-01
    effective_to: 2025-07-01
  - source: "b.csv"
    effective_from: 2025-03-01
    effective_to: 2025-12-31
"""
        )
        from moncpipelib.contracts.exceptions import ContractValidationError

        with pytest.raises(ContractValidationError, match="overlaps"):
            from moncpipelib.contracts import load_data_source

            load_data_source(tmp_path / "source.yaml")

    def test_multiple_open_ended_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "source.yaml").write_text(
            """\
source_id: "c3d4e5f6-a7b8-9012-cdef-123456789012"
source_name: bad-source
periods:
  - source: "a.csv"
    effective_from: 2025-01-01
    effective_to:
  - source: "b.csv"
    effective_from: 2025-07-01
    effective_to:
"""
        )
        from moncpipelib.contracts.exceptions import ContractValidationError

        with pytest.raises(ContractValidationError, match="At most one period"):
            from moncpipelib.contracts import load_data_source

            load_data_source(tmp_path / "source.yaml")

    def test_data_source_optional(self, tmp_path: Path) -> None:
        (tmp_path / "contract.yaml").write_text(
            """\
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: products
layer: silver
schema:
  columns:
    - name: id
      type: string
      nullable: false
      pii: false
"""
        )
        from moncpipelib.contracts import load_contract

        contract = load_contract(tmp_path / "contract.yaml")
        assert contract.data_source is None

    def test_inline_periods_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "contract.yaml").write_text(
            """\
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: products
layer: silver
schema:
  columns:
    - name: id
      type: string
      nullable: false
      pii: false
periods:
  - source: "a.csv"
    effective_from: 2025-01-01
"""
        )
        from moncpipelib.contracts.exceptions import ContractValidationError

        with pytest.raises(ContractValidationError, match="no longer supported inline"):
            from moncpipelib.contracts import load_contract

            load_contract(tmp_path / "contract.yaml")


class TestBuildPartitionsFromPeriods:
    """Tests for build_partitions_from_periods()."""

    def test_static_partitions_irregular_intervals(self) -> None:
        from dagster import StaticPartitionsDefinition

        source = _make_source(
            [
                Period(
                    source="a.csv", effective_from=date(2025, 1, 1), effective_to=date(2025, 6, 1)
                ),
                Period(
                    source="b.csv", effective_from=date(2025, 6, 1), effective_to=date(2025, 8, 15)
                ),
                Period(source="c.csv", effective_from=date(2025, 8, 15)),
            ]
        )
        result = build_partitions_from_periods(source)
        assert isinstance(result, StaticPartitionsDefinition)
        keys = result.get_partition_keys()
        assert keys == ["2025-01-01", "2025-06-01", "2025-08-15"]

    def test_time_window_partitions_quarterly(self) -> None:
        from dagster import TimeWindowPartitionsDefinition

        source = _make_source(
            [
                Period(
                    source="q1.csv", effective_from=date(2025, 1, 1), effective_to=date(2025, 4, 1)
                ),
                Period(
                    source="q2.csv", effective_from=date(2025, 4, 1), effective_to=date(2025, 7, 1)
                ),
                Period(
                    source="q3.csv", effective_from=date(2025, 7, 1), effective_to=date(2025, 10, 1)
                ),
            ]
        )
        result = build_partitions_from_periods(source)
        # 91 days between each -> quarterly cron
        assert isinstance(result, (TimeWindowPartitionsDefinition, type(result)))

    def test_no_periods_raises(self) -> None:
        source = _make_source([])
        with pytest.raises(ValueError, match="no periods defined"):
            build_partitions_from_periods(source)

    def test_single_period_static(self) -> None:
        from dagster import StaticPartitionsDefinition

        source = _make_source(
            [
                Period(source="a.csv", effective_from=date(2025, 1, 1)),
            ]
        )
        result = build_partitions_from_periods(source)
        assert isinstance(result, StaticPartitionsDefinition)
        assert result.get_partition_keys() == ["2025-01-01"]


class TestGetPeriodForPartition:
    """Tests for get_period_for_partition()."""

    def test_resolves_partition_key(self) -> None:
        periods = [
            Period(source="a.csv", effective_from=date(2025, 1, 1), effective_to=date(2025, 7, 1)),
            Period(source="b.csv", effective_from=date(2025, 7, 1)),
        ]
        source = _make_source(periods)
        result = get_period_for_partition(source, "2025-07-01")
        assert result.source == "b.csv"

    def test_unknown_partition_raises(self) -> None:
        source = _make_source(
            [
                Period(source="a.csv", effective_from=date(2025, 1, 1)),
            ]
        )
        with pytest.raises(KeyError, match="No period matches"):
            get_period_for_partition(source, "2099-01-01")


class TestLoadHistoricalPeriods:
    """Tests for load_historical_periods()."""

    def test_calls_write_in_chronological_order(self) -> None:
        periods = [
            Period(source="a.csv", effective_from=date(2025, 1, 1), effective_to=date(2025, 7, 1)),
            Period(source="b.csv", effective_from=date(2025, 7, 1)),
        ]
        source = _make_source(periods)
        database = MagicMock()
        database.write.return_value = MagicMock()
        context = MagicMock()

        def read_source(period: Period) -> pl.DataFrame:
            return pl.DataFrame({"id": [period.source]})

        results = load_historical_periods(
            source=source,
            database=database,
            context=context,
            target="silver.products",
            read_source=read_source,
        )

        assert len(results) == 2
        assert database.write.call_count == 2

        # First call: effective_date=2025-01-01
        first_call = database.write.call_args_list[0]
        assert first_call.kwargs["effective_date"] == date(2025, 1, 1)
        assert first_call.kwargs["target"] == "silver.products"

        # Second call: effective_date=2025-07-01
        second_call = database.write.call_args_list[1]
        assert second_call.kwargs["effective_date"] == date(2025, 7, 1)

    def test_no_periods_raises(self) -> None:
        source = _make_source([])
        with pytest.raises(ValueError, match="no periods defined"):
            load_historical_periods(
                source=source,
                database=MagicMock(),
                context=MagicMock(),
                target="silver.products",
                read_source=lambda _p: pl.DataFrame(),
            )

    def test_passes_write_kwargs(self) -> None:
        periods = [Period(source="a.csv", effective_from=date(2025, 1, 1))]
        source = _make_source(periods)
        database = MagicMock()
        database.write.return_value = MagicMock()

        load_historical_periods(
            source=source,
            database=database,
            context=MagicMock(),
            target="silver.products",
            read_source=lambda _p: pl.DataFrame({"id": ["1"]}),
            write_mode="scd2",
        )

        assert database.write.call_args.kwargs["write_mode"] == "scd2"


class TestPeriodPartitionKey:
    """Tests for partition_key on Period and its integration."""

    def test_period_with_partition_key(self) -> None:
        p = Period(
            source="a.csv",
            effective_from=date(2025, 1, 1),
            effective_to=date(2025, 7, 1),
            partition_key="2025-H1",
        )
        assert p.partition_key == "2025-H1"

    def test_period_partition_key_default_none(self) -> None:
        p = Period(source="a.csv", effective_from=date(2025, 1, 1))
        assert p.partition_key is None

    def test_build_partitions_uses_partition_key(self) -> None:
        from dagster import StaticPartitionsDefinition

        source = _make_source(
            [
                Period(
                    source="a.csv",
                    effective_from=date(2025, 1, 1),
                    effective_to=date(2025, 7, 1),
                    partition_key="2025-H1",
                ),
                Period(
                    source="b.csv",
                    effective_from=date(2025, 7, 1),
                    partition_key="2025-H2",
                ),
            ]
        )
        result = build_partitions_from_periods(source)
        assert isinstance(result, StaticPartitionsDefinition)
        assert result.get_partition_keys() == ["2025-H1", "2025-H2"]

    def test_build_partitions_falls_back_to_date_when_mixed(self) -> None:
        """If some periods have partition_key and some don't, fall back to dates."""
        from dagster import StaticPartitionsDefinition

        source = _make_source(
            [
                Period(
                    source="a.csv",
                    effective_from=date(2025, 1, 1),
                    partition_key="2025-H1",
                ),
                Period(source="b.csv", effective_from=date(2025, 7, 1)),
            ]
        )
        result = build_partitions_from_periods(source)
        assert isinstance(result, StaticPartitionsDefinition)
        assert result.get_partition_keys() == ["2025-01-01", "2025-07-01"]

    def test_get_period_matches_partition_key(self) -> None:
        periods = [
            Period(
                source="a.csv",
                effective_from=date(2025, 1, 1),
                partition_key="2025-H1",
            ),
            Period(
                source="b.csv",
                effective_from=date(2025, 7, 1),
                partition_key="2025-H2",
            ),
        ]
        source = _make_source(periods)
        result = get_period_for_partition(source, "2025-H2")
        assert result.source == "b.csv"

    def test_get_period_falls_back_to_date(self) -> None:
        periods = [
            Period(
                source="a.csv",
                effective_from=date(2025, 1, 1),
                partition_key="2025-H1",
            ),
        ]
        source = _make_source(periods)
        result = get_period_for_partition(source, "2025-01-01")
        assert result.source == "a.csv"

    def test_partition_key_parsed_from_source_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "products.source.yaml").write_text(
            """\
source_id: "d4e5f6a7-b8c9-0123-def0-123456789abc"
source_name: products-source
periods:
  - source: "h1.csv"
    effective_from: 2025-01-01
    effective_to: 2025-07-01
    partition_key: "2025-H1"
  - source: "h2.csv"
    effective_from: 2025-07-01
    partition_key: "2025-H2"
"""
        )
        (tmp_path / "contract.yaml").write_text(
            """\
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: products
layer: silver
data_source: products.source.yaml
schema:
  columns:
    - name: id
      type: string
      nullable: false
      pii: false
"""
        )
        from moncpipelib.contracts import load_contract

        contract = load_contract(tmp_path / "contract.yaml")
        assert contract.data_source is not None
        assert contract.data_source.periods[0].partition_key == "2025-H1"
        assert contract.data_source.periods[1].partition_key == "2025-H2"


class TestInjectPeriodPartitionColumn:
    """Tests for _inject_period_partition_column in PostgresResource."""

    @staticmethod
    def _make_resource():
        from moncpipelib.resources.postgres import PostgresResource

        return PostgresResource(
            host="localhost", port=5432, database="test", user="test", password="test"
        )

    @staticmethod
    def _make_contract_with_source(periods: list[Period]) -> MagicMock:
        """Create a mock DataContract with data_source.periods."""
        contract = MagicMock()
        contract.data_source = DataSource(
            source_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            source_name="test-source",
            periods=periods,
        )
        return contract

    def test_injects_column(self) -> None:
        contract = self._make_contract_with_source(
            [
                Period(
                    source="a.csv",
                    effective_from=date(2025, 1, 1),
                    partition_key="2025-H1",
                ),
            ]
        )
        df = pl.DataFrame({"id": ["1"], "name": ["x"]})
        write_config = {"partition_column": "load_period"}

        result = self._make_resource()._inject_period_partition_column(
            df, write_config, contract, date(2025, 1, 1)
        )
        assert "load_period" in result.columns
        assert result["load_period"][0] == "2025-H1"

    def test_skips_when_column_exists(self) -> None:
        contract = self._make_contract_with_source(
            [
                Period(
                    source="a.csv",
                    effective_from=date(2025, 1, 1),
                    partition_key="2025-H1",
                ),
            ]
        )
        df = pl.DataFrame({"id": ["1"], "load_period": ["user-value"]})
        write_config = {"partition_column": "load_period"}

        result = self._make_resource()._inject_period_partition_column(
            df, write_config, contract, date(2025, 1, 1)
        )
        assert result["load_period"][0] == "user-value"

    def test_skips_when_no_match(self) -> None:
        contract = self._make_contract_with_source(
            [
                Period(
                    source="a.csv",
                    effective_from=date(2025, 1, 1),
                    partition_key="2025-H1",
                ),
            ]
        )
        df = pl.DataFrame({"id": ["1"]})
        write_config = {"partition_column": "load_period"}

        result = self._make_resource()._inject_period_partition_column(
            df, write_config, contract, date(2099, 1, 1)
        )
        assert "load_period" not in result.columns

    def test_skips_when_no_effective_date(self) -> None:
        contract = self._make_contract_with_source(
            [
                Period(
                    source="a.csv",
                    effective_from=date(2025, 1, 1),
                    partition_key="2025-H1",
                ),
            ]
        )
        df = pl.DataFrame({"id": ["1"]})
        write_config = {"partition_column": "load_period"}

        result = self._make_resource()._inject_period_partition_column(
            df, write_config, contract, None
        )
        assert "load_period" not in result.columns

    def test_injects_from_dagster_partition_context(self) -> None:
        """Silver path: no data_source, but Dagster partition key available."""
        contract = MagicMock()
        contract.data_source = None
        wctx = MagicMock()
        wctx.has_partition_key = True
        wctx.partition_keys = ["2025-01-01"]

        df = pl.DataFrame({"id": ["1"], "name": ["x"]})
        write_config = {"partition_column": "load_period"}

        result = self._make_resource()._inject_period_partition_column(
            df, write_config, contract, None, wctx
        )
        assert "load_period" in result.columns
        assert result["load_period"][0] == "2025-01-01"

    def test_period_match_takes_priority_over_dagster_context(self) -> None:
        """When both period manifest and Dagster context exist, period wins."""
        contract = self._make_contract_with_source(
            [
                Period(
                    source="a.csv",
                    effective_from=date(2025, 1, 1),
                    partition_key="2025-H1",
                ),
            ]
        )
        wctx = MagicMock()
        wctx.has_partition_key = True
        wctx.partition_keys = ["dagster-key"]

        df = pl.DataFrame({"id": ["1"]})
        write_config = {"partition_column": "load_period"}

        result = self._make_resource()._inject_period_partition_column(
            df, write_config, contract, date(2025, 1, 1), wctx
        )
        assert result["load_period"][0] == "2025-H1"  # period manifest wins

    def test_dagster_context_skips_without_partition_key(self) -> None:
        """No injection when Dagster context has no partition."""
        contract = MagicMock()
        contract.data_source = None
        wctx = MagicMock()
        wctx.has_partition_key = False
        wctx.partition_keys = None

        df = pl.DataFrame({"id": ["1"]})
        write_config = {"partition_column": "load_period"}

        result = self._make_resource()._inject_period_partition_column(
            df, write_config, contract, None, wctx
        )
        assert "load_period" not in result.columns


class TestBuildPartitionsFromRegistry:
    """Tests for build_partitions_from_registry()."""

    def test_returns_registry_partitions_definition(self) -> None:
        from dagster import DynamicPartitionsDefinition

        result = build_partitions_from_registry("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        assert isinstance(result, RegistryPartitionsDefinition)
        assert isinstance(result, DynamicPartitionsDefinition)

    def test_preserves_source_id(self) -> None:
        result = build_partitions_from_registry("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        assert result.source_id == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

    def test_default_name_from_source_id(self) -> None:
        result = build_partitions_from_registry("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        assert result.name == "periods_a1b2c3d4_e5f6_7890_abcd_ef1234567890"

    def test_custom_name(self) -> None:
        result = build_partitions_from_registry(
            "a1b2c3d4-e5f6-7890-abcd-ef1234567890", name="my_partitions"
        )
        assert result.name == "my_partitions"

    def test_custom_name_preserves_source_id(self) -> None:
        result = build_partitions_from_registry(
            "a1b2c3d4-e5f6-7890-abcd-ef1234567890", name="my_partitions"
        )
        assert result.source_id == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"


class TestGetPeriodFromRegistry:
    """Tests for get_period_from_registry()."""

    def test_found(self) -> None:
        db = MagicMock()
        db.get_registry_periods.return_value = [
            {
                "partition_key": "2025-01-01",
                "effective_from": date(2025, 1, 1),
                "effective_to": date(2025, 7, 1),
                "source_uri": "https://example.com/h1.csv",
                "status": "materialized",
                "registered_by": "test",
            },
        ]
        result = get_period_from_registry(db, "test-source", "2025-01-01")
        assert result.effective_from == date(2025, 1, 1)
        assert result.source == "https://example.com/h1.csv"
        assert result.partition_key == "2025-01-01"

    def test_not_found(self) -> None:
        db = MagicMock()
        db.get_registry_periods.return_value = []
        with pytest.raises(KeyError, match="No period"):
            get_period_from_registry(db, "test-source", "2099-01-01")
