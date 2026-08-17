"""Schema-validation tests for the match: many field on source contracts (#438)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

import pytest

from moncpipelib.contracts.loader import validate_data_source_schema

_UUID = "22222222-3333-4444-5555-666666666666"


def _source_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "source_id": _UUID,
        "source_name": "trilliant-bronze",
        "ingest_source": "trilliant-visits-oncology",
        "periods": {
            "mode": "from_ingest",
            "template": {
                "source": "*.parquet",
                "effective_from_field": "partition_key",
                "match": "many",
            },
        },
    }
    data.update(overrides)
    return data


def test_template_match_many_valid() -> None:
    assert validate_data_source_schema(_source_data()) == []


def test_template_match_invalid_rejected() -> None:
    data = _source_data()
    data["periods"]["template"]["match"] = "several"
    errors = validate_data_source_schema(data)
    assert any("periods.template.match" in e for e in errors)


def test_period_match_many_valid() -> None:
    data = {
        "source_id": _UUID,
        "source_name": "trilliant-bronze",
        "ingest_source": "trilliant-visits-oncology",
        "periods": [
            {
                "source": "*.parquet",
                "effective_from": date(2025, 1, 1),
                "partition_key": "202501",
                "match": "many",
            }
        ],
    }
    assert validate_data_source_schema(data) == []


def test_period_match_invalid_rejected() -> None:
    data = {
        "source_id": _UUID,
        "source_name": "trilliant-bronze",
        "ingest_source": "trilliant-visits-oncology",
        "periods": [
            {
                "source": "*.parquet",
                "effective_from": date(2025, 1, 1),
                "partition_key": "202501",
                "match": "nope",
            }
        ],
    }
    errors = validate_data_source_schema(data)
    assert any("match" in e for e in errors)
    # The message keeps the "Period N: ..." shape every sibling error in
    # this loop uses (#467). Routing through the shared enum helper must
    # not turn it into `'Period 0.match' ...`, which quotes a prose
    # prefix as if it were a dotted field path.
    (msg,) = [e for e in errors if "match" in e]
    assert msg == "Period 0: 'match' must be one of ['many', 'one'], got 'nope'"


# ---------------------------------------------------------------------------
# Shared _validate_enum_value routing (#464/#467): a mapping or list match
# value must produce exactly one error and never raise -- a bare
# `value not in KNOWN_MATCH_MODES` membership test calls hash(value) and
# crashes on these otherwise-legal YAML values.
# ---------------------------------------------------------------------------


def _bad_template_match(bad: object) -> list[str]:
    data = _source_data()
    data["periods"]["template"]["match"] = bad
    return validate_data_source_schema(data)


def _bad_period_match(bad: object) -> list[str]:
    data: dict[str, Any] = {
        "source_id": _UUID,
        "source_name": "trilliant-bronze",
        "ingest_source": "trilliant-visits-oncology",
        "periods": [
            {
                "source": "*.parquet",
                "effective_from": date(2025, 1, 1),
                "partition_key": "202501",
                "match": bad,
            }
        ],
    }
    return validate_data_source_schema(data)


@pytest.mark.parametrize("bad", [{"a": "b"}, ["a"]], ids=["mapping", "list"])
@pytest.mark.parametrize(
    "field_marker,build_errors",
    [
        ("periods.template.match", _bad_template_match),
        ("match", _bad_period_match),
    ],
    ids=["template_match", "period_match"],
)
def test_match_field_unhashable_value_rejected_cleanly(
    field_marker: str, build_errors: Callable[[object], list[str]], bad: object
) -> None:
    errors = build_errors(bad)
    matches = [e for e in errors if field_marker in e]
    assert len(matches) == 1
    assert "must be one of" in matches[0]
