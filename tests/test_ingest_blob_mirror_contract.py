"""Schema-validation tests for the blob_mirror ingest contract block (#437)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from moncpipelib.contracts import (
    load_ingest_contract,
    validate_ingest_contract_schema,
)

_UUID = "11111111-2222-3333-4444-555555555555"


def _blob_mirror_data(**overrides: object) -> dict[str, object]:
    """A valid confidential blob_mirror ingest contract."""
    data: dict[str, object] = {
        "source_id": _UUID,
        "source_name": "trilliant-visits-oncology",
        "sensitivity": "confidential",
        "data_owner": "vp-data-platform",
        "compliance_review": "SECURITY.md#trilliant",
        "ingest": {
            "pattern": "blob_mirror",
            "prefix": "trilliant/visits_oncology/{partition_key}",
            "blob_mirror": {
                "source": {
                    "account_url": "https://examplestorageacct.blob.core.windows.net",
                    "container": "delivery",
                    "object_prefix": "{partition_key}/visits_oncology",
                    "discovery_prefix": "",
                    "partition_pattern": r"^\d{6}$",
                },
                "credential": {
                    "secret_name": "trilliant-sp",
                    "tenant_id": "partner-tenant",
                    "client_id": "our-sp",
                },
                "object_glob": "*.parquet",
                "exclude_globs": ["_committed_*", "_started_*", "_SUCCESS"],
            },
        },
    }
    data.update(overrides)
    return data


def _block(data: dict[str, Any]) -> dict[str, Any]:
    return data["ingest"]["blob_mirror"]


def test_blob_mirror_happy_path_passes() -> None:
    assert validate_ingest_contract_schema(_blob_mirror_data()) == []


def test_blob_mirror_parses(tmp_path: Path) -> None:
    import yaml

    p = tmp_path / "trilliant.ingest.yaml"
    p.write_text(yaml.safe_dump(_blob_mirror_data()))
    contract = load_ingest_contract(p)
    assert contract.pattern == "blob_mirror"
    assert contract.pattern_config["source"]["container"] == "delivery"
    assert contract.pattern_config["object_glob"] == "*.parquet"


def test_blob_mirror_requires_block() -> None:
    data = _blob_mirror_data()
    del data["ingest"]["blob_mirror"]  # type: ignore[attr-defined]
    errors = validate_ingest_contract_schema(data)
    assert any("'ingest.blob_mirror' is required" in e for e in errors)


def test_blob_mirror_requires_source_fields() -> None:
    data = _blob_mirror_data()
    del _block(data)["source"]["account_url"]
    del _block(data)["source"]["object_prefix"]
    errors = validate_ingest_contract_schema(data)
    assert any("source.account_url" in e for e in errors)
    assert any("source.object_prefix" in e for e in errors)


def test_blob_mirror_unknown_source_key_rejected() -> None:
    data = _blob_mirror_data()
    _block(data)["source"]["bogus"] = "x"
    errors = validate_ingest_contract_schema(data)
    assert any("bogus" in e for e in errors)


def test_blob_mirror_bad_partition_pattern_regex() -> None:
    data = _blob_mirror_data()
    _block(data)["source"]["partition_pattern"] = "([unclosed"
    errors = validate_ingest_contract_schema(data)
    assert any("partition_pattern" in e and "regex" in e for e in errors)


def test_blob_mirror_credential_partial_rejected() -> None:
    """A credential block must carry the full SP triple when present."""
    data = _blob_mirror_data()
    _block(data)["credential"] = {"secret_name": "s"}  # missing ids
    errors = validate_ingest_contract_schema(data)
    assert any("credential.tenant_id" in e for e in errors)
    assert any("credential.client_id" in e for e in errors)


def test_blob_mirror_credential_optional() -> None:
    """Local-dev contracts may omit credential (DefaultAzureCredential)."""
    data = _blob_mirror_data()
    del _block(data)["credential"]
    assert validate_ingest_contract_schema(data) == []


def test_blob_mirror_exclude_globs_type_checked() -> None:
    data = _blob_mirror_data()
    _block(data)["exclude_globs"] = "_SUCCESS"  # should be a list
    errors = validate_ingest_contract_schema(data)
    assert any("exclude_globs" in e for e in errors)
