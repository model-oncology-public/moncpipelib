"""Cookbook examples for code_hash() versioning utility."""

from __future__ import annotations

import pytest


@pytest.mark.cookbook(
    title="Automatic Code Versioning for Dagster Assets",
    description=(
        "Use code_hash() to automatically generate a code_version for Dagster "
        "assets. This enables stale/fresh detection in the Dagster UI -- assets "
        "whose logic hasn't changed since last materialization show as 'fresh'."
    ),
    category="versioning",
)
def test_code_hash_dagster_integration() -> None:
    """Demonstrate code_hash() for Dagster code_version."""
    # --- cookbook:start ---
    from moncpipelib import code_hash

    # Generate a code version for the current module
    version = code_hash()
    print(f"Code version: {version}")
    print(f"Length: {len(version)} characters")

    # Use with Dagster's @asset decorator:
    #
    #   @asset(code_version=code_hash())
    #   def my_asset(context):
    #       ...
    #
    # The hash includes:
    # - This Python source file
    # - Any *.contract.yaml files in the same directory
    #
    # When either changes, the hash changes, and Dagster marks
    # the asset as stale (needs re-materialization).

    # The hash is deterministic - same input always produces same output
    assert code_hash() == version
    print(f"Deterministic: {code_hash() == version}")
    # --- cookbook:end ---

    assert len(version) == 12
    assert all(c in "0123456789abcdef" for c in version)
