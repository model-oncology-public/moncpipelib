"""Cookbook tests for contract parameters.

Each test here doubles as a documentation example. The code between
``# --- cookbook:start ---`` and ``# --- cookbook:end ---`` markers is
extracted by the cookbook pytest plugin and rendered into docs/cookbook.md.
"""

from __future__ import annotations

import pytest

from moncpipelib.contracts.models import (
    Column,
    ColumnType,
    DataContract,
    Schema,
)


def _contract_with_parameters() -> DataContract:
    """Build a contract with business-logic parameters."""
    return DataContract(
        version="1.0",
        pipeline_id="550e8400-e29b-41d4-a716-446655440000",
        asset="claims_silver",
        layer="silver",
        schema=Schema(
            columns=[
                Column(
                    name="claim_id",
                    type=ColumnType.STRING,
                    nullable=False,
                    primary_key=True,
                    pii=False,
                ),
                Column(
                    name="claim_date",
                    type=ColumnType.DATE,
                    nullable=False,
                    pii=False,
                ),
            ]
        ),
        parameters={
            "days_of_tolerance": 30,
            "include_expired": False,
            "allowed_statuses": ["approved", "pending"],
        },
    )


@pytest.mark.cookbook(
    title="Reading Business-Logic Parameters from a Contract",
    description=(
        "Use the `parameters:` section in a contract YAML to store "
        "business-logic configuration values. Access them via "
        "`contract.get_parameter()` which provides helpful error messages "
        "on typos or missing configuration."
    ),
    category="contracts",
)
def test_get_parameter_from_contract() -> None:
    """Demonstrate reading typed parameters from a contract."""
    contract = _contract_with_parameters()

    # --- cookbook:start ---
    # Access parameters defined in the contract YAML:
    #
    #   parameters:
    #     days_of_tolerance: 30
    #     include_expired: false
    #     allowed_statuses: ["approved", "pending"]

    tolerance = contract.get_parameter("days_of_tolerance")
    include_expired = contract.get_parameter("include_expired")
    statuses = contract.get_parameter("allowed_statuses")

    print(f"Days of tolerance: {tolerance}")
    print(f"Include expired:   {include_expired}")
    print(f"Allowed statuses:  {statuses}")
    # --- cookbook:end ---

    assert tolerance == 30
    assert include_expired is False
    assert statuses == ["approved", "pending"]


@pytest.mark.cookbook(
    title="Parameter Defaults and Error Handling",
    description=(
        "When a parameter is missing, `get_parameter()` raises a `KeyError` "
        "with context: it lists available parameters (suggesting a typo) or "
        "notes that no parameters section exists. Pass a default value to "
        "fall back gracefully -- a warning is logged with the default used."
    ),
    category="contracts",
)
def test_parameter_defaults_and_errors() -> None:
    """Demonstrate default values and error handling for parameters."""
    contract = _contract_with_parameters()

    # --- cookbook:start ---
    # Provide a default for optional parameters -- logs a warning
    max_retries = contract.get_parameter("max_retries", 3)
    print(f"Max retries (default): {max_retries}")

    # Missing parameter without a default raises KeyError with context
    try:
        contract.get_parameter("days_of_tolerence")  # typo!
    except KeyError as e:
        print(f"Error message: {e}")
        # KeyError includes available parameters and suggests checking for typos
    # --- cookbook:end ---

    assert max_retries == 3
    with pytest.raises(KeyError, match="typo"):
        contract.get_parameter("days_of_tolerence")
