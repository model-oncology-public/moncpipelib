"""Schema-validation tests for the match: many field on source contracts (#438)."""

from __future__ import annotations

from datetime import date
from typing import Any

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
