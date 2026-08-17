"""Contract-driven query builder for integration testing.

Builds SQL queries from contract definitions, automatically switching
between production and test table references based on environment variables.
"""

from __future__ import annotations

import os

from moncpipelib.contracts.loader import load_contract_for_asset
from moncpipelib.contracts.models import DataContract, TableReference


class AssetQueryBuilder:
    """Builds SQL queries from contract definitions.

    Automatically switches between production and test table references
    based on environment variables.

    Environment Variables:
        ONCALYTICS_TEST_SCHEMA: Test schema name (e.g., 'integration_tests').
            When set, queries use this schema instead of the contract's source schema.
        ONCALYTICS_TEST_TABLE_PREFIX: Prefix for test tables (e.g., 'johndoe_abc123_').
            Defaults to empty string.

    Example:
        # Production mode:
        builder = AssetQueryBuilder("fda_ndc_package_silver", "silver")
        sql = builder.get_source_query()
        # -> "SELECT * FROM synthetic_bronze.fda_ndc_package_raw"

        # Test mode (with env vars set):
        # ONCALYTICS_TEST_SCHEMA=integration_tests
        # ONCALYTICS_TEST_TABLE_PREFIX=johndoe_abc123_
        sql = builder.get_source_query()
        # -> "SELECT * FROM integration_tests.johndoe_abc123_synthetic_bronze_fda_ndc_package_raw_source"
    """

    def __init__(self, asset_name: str, layer: str) -> None:
        """Initialize the query builder.

        Args:
            asset_name: Dagster asset name
            layer: Data layer (bronze/silver/gold)
        """
        self.asset_name = asset_name
        self.layer = layer
        self._contract: DataContract | None = load_contract_for_asset(asset_name, layer=layer)

        # Check for test mode
        self.test_schema: str | None = os.getenv("ONCALYTICS_TEST_SCHEMA")
        self.test_prefix: str = os.getenv("ONCALYTICS_TEST_TABLE_PREFIX", "")

    @property
    def is_test_mode(self) -> bool:
        """Whether running in test mode (test schema is set)."""
        return self.test_schema is not None

    def get_source_query(
        self,
        columns: list[str] | None = None,
    ) -> str:
        """Build SELECT query for source table.

        In production: SELECT * FROM synthetic_bronze.fda_ndc_package_raw
        In testing: SELECT * FROM integration_tests.{prefix}synthetic_bronze_fda_ndc_package_raw_source

        Args:
            columns: Columns to select (default: *)

        Returns:
            SQL query string

        Raises:
            ValueError: If contract is missing or has no source tables
        """
        if self._contract is None:
            raise ValueError(
                f"No contract found for asset '{self.asset_name}' in layer '{self.layer}'"
            )

        source_tables = self._contract.get_source_tables()
        if not source_tables:
            raise ValueError(f"Contract for '{self.asset_name}' has no source tables defined")

        source = source_tables[0]  # Use first source table
        columns_str = ", ".join(columns) if columns else "*"

        if self.is_test_mode:
            # Build test table reference:
            # {test_prefix}{source_schema}_{source_table}_source
            test_table = f"{self.test_prefix}{source.schema}_{source.table}_source"
            return f"SELECT {columns_str} FROM {self.test_schema}.{test_table}"
        else:
            return f"SELECT {columns_str} FROM {source.schema_qualified_name}"

    def get_source_table_reference(self) -> TableReference:
        """Get the actual (production) source table reference.

        Returns the REAL source table location, not the test table.
        Used by integration test runner to know what to copy.

        Returns:
            TableReference for the first source table

        Raises:
            ValueError: If contract is missing or has no source tables
        """
        if self._contract is None:
            raise ValueError(
                f"No contract found for asset '{self.asset_name}' in layer '{self.layer}'"
            )

        source_tables = self._contract.get_source_tables()
        if not source_tables:
            raise ValueError(f"Contract for '{self.asset_name}' has no source tables defined")

        return source_tables[0]
