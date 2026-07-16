"""Cookbook tests for contract column test definitions and validation.

Shows every supported YAML format for defining column tests in data contracts,
how they are parsed, and how they validate DataFrames at write time.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.cookbook(
    title="Column Tests: Simple String Form",
    description=(
        "The simplest column tests are bare strings -- ``not_null``, ``unique``, "
        "and ``not_in_future``. These require no parameters."
    ),
    category="contracts",
)
def test_simple_string_tests(tmp_path: Path) -> None:
    """Demonstrate simple string-form column tests."""
    (tmp_path / "contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: events
layer: bronze
schema:
  columns:
    - name: event_id
      type: string
      nullable: false
      pii: false
      tests:
        - not_null
        - unique
    - name: event_date
      type: datetime
      nullable: false
      pii: false
      tests:
        - not_in_future
"""
    )

    # --- cookbook:start ---
    import polars as pl

    from moncpipelib.contracts import load_contract
    from moncpipelib.contracts.validators import run_column_test

    contract = load_contract(tmp_path / "contract.yaml")

    # Simple tests have no parameters
    for col in contract.schema.columns:
        for test in col.tests:
            print(f"{col.name}.{test.test_type}: params={test.parameters}")

    # Validate against a DataFrame
    df = pl.DataFrame(
        {
            "event_id": ["E-001", "E-002", "E-003"],
            "event_date": ["2025-01-15", "2025-06-01", "2025-12-31"],
        }
    ).with_columns(pl.col("event_date").str.to_datetime())

    result = run_column_test(df, "event_id", "not_null", {})
    print(f"\nnot_null check: {'PASSED' if result.passed else 'FAILED'}")
    result = run_column_test(df, "event_id", "unique", {})
    print(f"unique check: {'PASSED' if result.passed else 'FAILED'}")
    # --- cookbook:end ---

    assert result.passed


@pytest.mark.cookbook(
    title="Column Types: JSON and JSONB",
    description=(
        "Contracts support ``json`` and ``jsonb`` column types for PostgreSQL JSON "
        "columns. Both are represented as strings in Polars (JSON text). String-based "
        "column tests like ``min_length`` and ``pattern`` work on JSON columns since "
        "they are stored as text."
    ),
    category="contracts",
)
def test_json_column_types(tmp_path: Path) -> None:
    """Demonstrate json and jsonb column types in contracts."""
    (tmp_path / "contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: api_responses
layer: bronze
schema:
  columns:
    - name: request_id
      type: string
      nullable: false
      pii: false
    - name: response_body
      type: jsonb
      nullable: true
      pii: false
      tests:
        - min_length: 2
    - name: request_headers
      type: json
      nullable: true
      pii: false
"""
    )

    # --- cookbook:start ---
    import polars as pl

    from moncpipelib.contracts import ColumnType, load_contract
    from moncpipelib.contracts.validators import validate_schema

    contract = load_contract(tmp_path / "contract.yaml")

    # Both json and jsonb are first-class contract types
    for col in contract.schema.columns:
        print(f"{col.name}: type={col.type.value}")

    # JSON/JSONB columns are String in Polars -- no special conversion needed
    df = pl.DataFrame(
        {
            "request_id": ["req-001", "req-002"],
            "response_body": ['{"status": "ok", "data": [1, 2]}', '{"status": "error"}'],
            "request_headers": ['{"Content-Type": "application/json"}', None],
        }
    )

    result = validate_schema(df, contract)
    print(f"\nSchema validation: {'PASSED' if result.passed else 'FAILED'}")

    # The parsed column types are ColumnType.JSONB and ColumnType.JSON
    resp_col = contract.get_column("response_body")
    assert resp_col is not None
    print(f"response_body type enum: {resp_col.type}")
    assert resp_col.type == ColumnType.JSONB
    # --- cookbook:end ---

    assert result.passed


@pytest.mark.cookbook(
    title="Column Tests: Comparison Operators",
    description=(
        "Comparison tests (``greater_than``, ``less_than_or_equal``, etc.) use "
        "the shorthand dict form where the test type is the key and the "
        'threshold is the value. This maps to ``parameters={"value": N}``.'
    ),
    category="contracts",
)
def test_comparison_tests(tmp_path: Path) -> None:
    """Demonstrate comparison operator column tests."""
    (tmp_path / "contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: metrics
layer: silver
schema:
  columns:
    - name: score
      type: integer
      nullable: false
      pii: false
      tests:
        - greater_than_or_equal: 0
        - less_than_or_equal: 100
    - name: temperature
      type: decimal
      nullable: false
      pii: false
      tests:
        - greater_than: -273.15
        - less_than: 1000
"""
    )

    # --- cookbook:start ---
    import polars as pl

    from moncpipelib.contracts import load_contract
    from moncpipelib.contracts.validators import run_column_test

    contract = load_contract(tmp_path / "contract.yaml")

    # Comparison tests store the threshold under "value"
    for col in contract.schema.columns:
        for test in col.tests:
            print(f"{col.name}.{test.test_type}: params={test.parameters}")

    # Validate
    df = pl.DataFrame({"score": [0, 50, 100], "temperature": [36.6, 98.6, 212.0]})

    result = run_column_test(df, "score", "greater_than_or_equal", {"value": 0})
    print(f"\nscore >= 0: {'PASSED' if result.passed else 'FAILED'}")

    result = run_column_test(df, "score", "less_than_or_equal", {"value": 100})
    print(f"score <= 100: {'PASSED' if result.passed else 'FAILED'}")
    # --- cookbook:end ---

    assert result.passed


@pytest.mark.cookbook(
    title="Column Tests: Between Range",
    description=(
        "The ``between`` test uses a dict parameter form with ``min`` and ``max`` "
        "keys. In YAML this is written as a nested mapping under the test type."
    ),
    category="contracts",
)
def test_between_test(tmp_path: Path) -> None:
    """Demonstrate between range test."""
    (tmp_path / "contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: measurements
layer: bronze
schema:
  columns:
    - name: percentage
      type: decimal
      nullable: false
      pii: false
      tests:
        - between:
            min: 0
            max: 100
"""
    )

    # --- cookbook:start ---
    import polars as pl

    from moncpipelib.contracts import load_contract
    from moncpipelib.contracts.validators import run_column_test

    contract = load_contract(tmp_path / "contract.yaml")
    col = contract.schema.columns[0]
    test = col.tests[0]

    # Dict parameters are passed through directly
    print(f"{col.name}.{test.test_type}: params={test.parameters}")

    df = pl.DataFrame({"percentage": [0.0, 25.5, 50.0, 99.9]})
    result = run_column_test(df, "percentage", "between", {"min": 0, "max": 100})
    print(f"between 0-100: {'PASSED' if result.passed else 'FAILED'}")
    # --- cookbook:end ---

    assert result.passed


@pytest.mark.cookbook(
    title="Column Tests: Accepted Values",
    description=(
        "Define allowed values for a column with ``accepted_values``. "
        "Two YAML formats are supported: a flat list directly under the "
        "test key (recommended), or a nested ``values`` mapping. Both "
        "produce the same validation behavior."
    ),
    category="contracts",
)
def test_accepted_values_formats(tmp_path: Path) -> None:
    """Demonstrate both accepted_values YAML formats."""
    # Format 1: flat list (recommended, natural YAML)
    (tmp_path / "contract_list.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: calendar
layer: gold
schema:
  columns:
    - name: quarter
      type: string
      nullable: false
      pii: false
      tests:
        - accepted_values:
            - Q1
            - Q2
            - Q3
            - Q4
"""
    )

    # Format 2: nested dict with "values" key
    (tmp_path / "contract_dict.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "b2c3d4e5-f6a7-8901-bcde-f12345678901"
asset: orders
layer: bronze
schema:
  columns:
    - name: status
      type: string
      nullable: false
      pii: false
      tests:
        - accepted_values:
            values:
              - pending
              - approved
              - denied
"""
    )

    # --- cookbook:start ---
    import polars as pl

    from moncpipelib.contracts import load_contract
    from moncpipelib.contracts.validators import run_column_test

    # Format 1: flat list (recommended)
    contract1 = load_contract(tmp_path / "contract_list.yaml")
    test1 = contract1.schema.columns[0].tests[0]
    print("Flat list format:")
    print(f"  test_type: {test1.test_type}")
    print(f"  parameters: {test1.parameters}")

    # Format 2: nested dict with "values" key
    contract2 = load_contract(tmp_path / "contract_dict.yaml")
    test2 = contract2.schema.columns[0].tests[0]
    print("\nNested dict format:")
    print(f"  test_type: {test2.test_type}")
    print(f"  parameters: {test2.parameters}")

    # Both formats validate correctly
    df = pl.DataFrame({"quarter": ["Q1", "Q3", "Q4"]})

    # Flat list: values stored under "value" key
    result1 = run_column_test(df, "quarter", test1.test_type, test1.parameters)
    print(f"\nFlat list validation: {'PASSED' if result1.passed else 'FAILED'}")

    # Dict form: values stored under "values" key
    df2 = pl.DataFrame({"status": ["pending", "approved"]})
    result2 = run_column_test(df2, "status", test2.test_type, test2.parameters)
    print(f"Dict form validation: {'PASSED' if result2.passed else 'FAILED'}")
    # --- cookbook:end ---

    assert result1.passed
    assert result2.passed


@pytest.mark.cookbook(
    title="Column Tests: Pattern Matching",
    description=(
        "The ``pattern`` test validates column values against a regular expression. "
        "Use ``max_length`` and ``min_length`` for simple string length constraints."
    ),
    category="contracts",
)
def test_pattern_and_length_tests(tmp_path: Path) -> None:
    """Demonstrate pattern and length column tests."""
    (tmp_path / "contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: patients
layer: silver
schema:
  columns:
    - name: patient_id
      type: string
      nullable: false
      pii: true
      tests:
        - pattern: "^PAT-[0-9]{8}$"
    - name: state_code
      type: string
      nullable: false
      pii: false
      tests:
        - min_length: 2
        - max_length: 2
"""
    )

    # --- cookbook:start ---
    import polars as pl

    from moncpipelib.contracts import load_contract
    from moncpipelib.contracts.validators import run_column_test

    contract = load_contract(tmp_path / "contract.yaml")

    for col in contract.schema.columns:
        for test in col.tests:
            print(f"{col.name}.{test.test_type}: params={test.parameters}")

    # Pattern validation
    df = pl.DataFrame(
        {
            "patient_id": ["PAT-00000001", "PAT-12345678"],
            "state_code": ["TX", "CA"],
        }
    )
    result = run_column_test(df, "patient_id", "pattern", {"value": "^PAT-[0-9]{8}$"})
    print(f"\npattern check: {'PASSED' if result.passed else 'FAILED'}")

    result = run_column_test(df, "state_code", "max_length", {"value": 2})
    print(f"max_length check: {'PASSED' if result.passed else 'FAILED'}")
    # --- cookbook:end ---

    assert result.passed


@pytest.mark.cookbook(
    title="Column Tests: Severity and Conditional Modifiers",
    description=(
        "Add ``severity: warn`` to downgrade a test failure from an error to a "
        "warning. Use ``when: not_null`` to skip validation for null values, "
        "useful for nullable columns where the test only applies to present data."
    ),
    category="contracts",
)
def test_severity_and_when_modifiers(tmp_path: Path) -> None:
    """Demonstrate severity and when modifiers on column tests."""
    (tmp_path / "contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: claims
layer: silver
schema:
  columns:
    - name: amount
      type: decimal
      nullable: false
      pii: false
      tests:
        - greater_than: 0
          severity: warn
    - name: npi
      type: string
      nullable: true
      pii: false
      tests:
        - pattern: "^[0-9]{10}$"
          when: not_null
          severity: error
"""
    )

    # --- cookbook:start ---
    import polars as pl

    from moncpipelib.contracts import Severity, load_contract
    from moncpipelib.contracts.validators import run_column_test

    contract = load_contract(tmp_path / "contract.yaml")

    for col in contract.schema.columns:
        for test in col.tests:
            print(f"{col.name}.{test.test_type}: severity={test.severity.value}, when={test.when}")

    # severity=warn means validation failure is logged but doesn't block writes
    amount_test = contract.schema.columns[0].tests[0]
    print(f"\namount test severity: {amount_test.severity}")
    assert amount_test.severity == Severity.WARN

    # when=not_null filters to non-null rows before testing
    # Null NPI values are skipped, only present values must match the pattern
    df = pl.DataFrame({"npi": ["1234567890", None, "0987654321"]})
    npi_test = contract.schema.columns[1].tests[0]

    result = run_column_test(
        df.filter(pl.col("npi").is_not_null()),  # when: not_null
        "npi",
        npi_test.test_type,
        npi_test.parameters,
    )
    print(f"npi pattern (non-null only): {'PASSED' if result.passed else 'FAILED'}")
    # --- cookbook:end ---

    assert result.passed


@pytest.mark.cookbook(
    title="Column Tests: Rejected Values (not_in)",
    description=(
        "The ``not_in`` test is the inverse of ``accepted_values`` -- it specifies "
        "a blacklist of values that should not appear in the column. Useful for "
        "sentinel values (e.g., -1 for no match) or placeholder strings. "
        "Null values are ignored. Combine with ``severity: warn`` to surface "
        "data quality issues without blocking writes."
    ),
    category="contracts",
)
def test_not_in_rejected_values(tmp_path: Path) -> None:
    """Demonstrate not_in (rejected values) column test."""
    (tmp_path / "contract.yaml").write_text(
        """\
version: "1.0"
pipeline_id: "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
asset: matches
layer: silver
schema:
  strict: false
  columns:
    - name: match_score
      type: integer
      nullable: false
      pii: false
      tests:
        - not_in:
            values: [-1, -999]
          severity: warn
    - name: source
      type: string
      nullable: true
      pii: false
      tests:
        - not_in:
            values: ["UNKNOWN", "N/A", "TBD"]
"""
    )

    # --- cookbook:start ---
    import polars as pl

    from moncpipelib.contracts import load_contract
    from moncpipelib.contracts.validators import run_column_test

    contract = load_contract(tmp_path / "contract.yaml")

    # not_in tests define a blacklist of rejected values
    for col in contract.schema.columns:
        for test in col.tests:
            print(f"{col.name}.{test.test_type}: rejected={test.parameters}")

    # DataFrame with some rejected sentinel values
    df = pl.DataFrame(
        {
            "match_score": [85, -1, 92, 100, -999],
            "source": ["EMR", "UNKNOWN", "Claims", None, "N/A"],
        }
    )

    # Check match_score for sentinels
    result = run_column_test(df, "match_score", "not_in", {"values": [-1, -999]})
    print(f"\nmatch_score not_in [-1, -999]: {'PASSED' if result.passed else 'FAILED'}")
    print(f"  Rows with rejected values: {result.failed_count}")

    # Check source for placeholder strings
    result2 = run_column_test(df, "source", "not_in", {"values": ["UNKNOWN", "N/A", "TBD"]})
    print(f"source not_in placeholders: {'PASSED' if result2.passed else 'FAILED'}")
    print(f"  Rows with rejected values: {result2.failed_count}")

    # Null values are ignored -- only non-null matches count
    print("  (null source values not counted as rejected)")
    # --- cookbook:end ---

    assert result.passed is False
    assert result.failed_count == 2
    assert result2.passed is False
    assert result2.failed_count == 2
