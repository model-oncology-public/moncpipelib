"""Cookbook examples for downstream integration-test helpers."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from moncpipelib.contracts.models import (
    Column,
    ColumnType,
    DataContract,
    Schema,
)


@pytest.mark.cookbook(
    title="Safe WHERE Clause Parameterization",
    description=(
        "``SafeWhereClauseBuilder`` lets integration-test harnesses accept "
        "free-form WHERE clauses (e.g. from test config) without opening a SQL "
        "injection hole. It rejects dangerous keywords (DROP, DELETE, UNION, "
        "...), parses the clause with sqlglot to validate structure, optionally "
        "enforces a column allowlist, and rewrites string/number literals into "
        "psycopg ``%(param)s`` placeholders with a matching params dict -- so "
        "the value never touches the SQL text. This is the right tool when a "
        "WHERE clause is data-driven; never f-string user input into SQL."
    ),
    category="testing",
)
def test_safe_where_clause_builder() -> None:
    """Demonstrate validating and parameterizing a WHERE clause."""
    # --- cookbook:start ---
    from moncpipelib import SafeWhereClauseBuilder, SQLSafetyError

    builder = SafeWhereClauseBuilder(allowed_columns=["status", "created_date"])

    # Literals are extracted into parameters, not interpolated into SQL
    clause, params = builder.validate_and_parameterize(
        "status = 'active' AND created_date >= '2024-01-01'"
    )
    print("=== Parameterized clause ===")
    print("clause:", clause)
    print("params:", params)

    # A column outside the allowlist is rejected
    print()
    print("=== Allowlist enforcement ===")
    try:
        builder.validate_and_parameterize("secret_col = 'x'")
    except SQLSafetyError as exc:
        print("rejected:", exc)

    # An injection attempt via a dangerous keyword is rejected
    try:
        builder.validate_and_parameterize("1=1; DROP TABLE patients")
    except SQLSafetyError as exc:
        print("rejected:", exc)
    # --- cookbook:end ---

    assert "%(param_0)s" in clause
    assert "%(param_1)s" in clause
    assert set(params.values()) == {"active", "2024-01-01"}
    with pytest.raises(SQLSafetyError):
        builder.validate_and_parameterize("secret_col = 'x'")
    with pytest.raises(SQLSafetyError):
        builder.validate_and_parameterize("1=1; DROP TABLE patients")


def _sample_contract() -> DataContract:
    """A minimal contract used as test scaffolding for the example below.

    In real usage the contract is loaded from disk by the asset name; here we
    construct one inline and patch the loader so the example stays focused on
    the env-var-driven query switching.
    """
    return DataContract(
        version="1.0",
        pipeline_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        asset="fda_ndc_package_silver",
        layer="silver",
        schema=Schema(
            columns=[
                Column(name="id", type=ColumnType.INTEGER, nullable=False),
                Column(name="ndc", type=ColumnType.STRING, nullable=True),
            ]
        ),
        sources=[
            {
                "type": "table",
                "database": "analytics",
                "schema": "synthetic_bronze",
                "table": "fda_ndc_package_raw",
            }
        ],
    )


@pytest.mark.cookbook(
    title="Contract-Driven Query Building: Prod vs Test Tables",
    description=(
        "``AssetQueryBuilder`` reads an asset's data contract and builds the "
        "SELECT against its source table. Its value is environment-driven "
        "redirection: with no env vars it targets the real production source "
        "table, but when ``ONCALYTICS_TEST_SCHEMA`` (and optionally "
        "``ONCALYTICS_TEST_TABLE_PREFIX``) are set, the identical call instead "
        "targets a per-developer test table in the test schema. This lets one "
        "set of asset code run unchanged against synthetic copies during "
        "integration tests. ``get_source_table_reference()`` always returns the "
        "real source location so the test runner knows what to copy."
    ),
    category="testing",
)
@patch("moncpipelib.testing.query_builder.load_contract_for_asset")
def test_asset_query_builder_env_switching(mock_load: MagicMock) -> None:
    """Demonstrate production vs. test query switching via env vars."""
    mock_load.return_value = _sample_contract()
    saved = {
        k: os.environ.get(k) for k in ("ONCALYTICS_TEST_SCHEMA", "ONCALYTICS_TEST_TABLE_PREFIX")
    }
    try:
        # --- cookbook:start ---
        # The test schema/prefix come from the environment (set by CI or a
        # shell); here we set them inline to show how they drive the switch.
        from moncpipelib import AssetQueryBuilder

        # Production mode: no test env vars -> query hits the real source table
        os.environ.pop("ONCALYTICS_TEST_SCHEMA", None)
        os.environ.pop("ONCALYTICS_TEST_TABLE_PREFIX", None)

        builder = AssetQueryBuilder("fda_ndc_package_silver", layer="silver")
        print("is_test_mode:", builder.is_test_mode)
        print("prod query:", builder.get_source_query())

        # Test mode: point the same asset at a per-developer synthetic copy
        os.environ["ONCALYTICS_TEST_SCHEMA"] = "integration_tests"
        os.environ["ONCALYTICS_TEST_TABLE_PREFIX"] = "johndoe_abc123_"

        builder = AssetQueryBuilder("fda_ndc_package_silver", layer="silver")
        print("is_test_mode:", builder.is_test_mode)
        print("test query: ", builder.get_source_query(columns=["id", "ndc"]))

        # The real source location is always available for the test runner to copy
        ref = builder.get_source_table_reference()
        print("source to copy:", ref.schema_qualified_name)
        # --- cookbook:end ---

        assert ref.schema == "synthetic_bronze"
        assert ref.table == "fda_ndc_package_raw"
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
