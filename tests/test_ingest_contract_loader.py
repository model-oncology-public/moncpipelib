"""Tests for load_ingest_contract + validate_ingest_contract_schema."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from moncpipelib.contracts import (
    ContractNotFoundError,
    ContractValidationError,
    IngestContract,
    load_ingest_contract,
    validate_ingest_contract_schema,
)

_UUID = "11111111-2222-3333-4444-555555555555"


def _happy_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "source_id": _UUID,
        "source_name": "cms-asp",
        "sensitivity": "public",
        "description": "CMS ASP quarterly releases",
        "ingest": {
            "pattern": "http_urls",
            "prefix": "cms_asp/{partition_key}",
            "extract": ["zip"],
            "strip_extensions": [".xls", ".xlsx"],
            "http_urls": {
                "idempotency": "hash_compare",
                "fetch": {"retries": 3, "timeout_s": 120},
                "periods": [
                    {
                        "partition_key": "2024-01-01",
                        "urls": ["https://example.com/a.zip"],
                    }
                ],
            },
        },
    }
    data.update(overrides)
    return data


def test_happy_path_parses(tmp_path: Path) -> None:
    import yaml

    p = tmp_path / "cms_asp.ingest.yaml"
    p.write_text(yaml.safe_dump(_happy_data()))

    contract = load_ingest_contract(p)

    assert isinstance(contract, IngestContract)
    assert contract.source_name == "cms-asp"
    assert contract.sensitivity == "public"
    assert contract.pattern == "http_urls"
    assert contract.prefix_template == "cms_asp/{partition_key}"
    assert contract.extract == ("zip",)
    assert contract.strip_extensions == (".xls", ".xlsx")
    assert contract.pattern_config["periods"][0]["partition_key"] == "2024-01-01"


def test_missing_file_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(ContractNotFoundError):
        load_ingest_contract(tmp_path / "nope.ingest.yaml")


def test_empty_file_rejected(tmp_path: Path) -> None:
    p = tmp_path / "x.ingest.yaml"
    p.write_text("")
    with pytest.raises(ContractValidationError, match="Empty"):
        load_ingest_contract(p)


def test_phi_without_attestation_fails() -> None:
    errors = validate_ingest_contract_schema(_happy_data(sensitivity="phi"))
    assert any("data_owner" in e for e in errors)
    assert any("compliance_review" in e for e in errors)


def test_confidential_with_attestation_passes() -> None:
    errors = validate_ingest_contract_schema(
        _happy_data(
            sensitivity="confidential",
            data_owner="data-platform-team",
            compliance_review="SECURITY.md#cms-asp",
        )
    )
    assert errors == []


def test_unknown_top_level_key_rejected() -> None:
    data = _happy_data()
    data["mystery"] = "x"
    errors = validate_ingest_contract_schema(data)
    assert any("mystery" in e for e in errors)


def test_bad_sensitivity_rejected() -> None:
    errors = validate_ingest_contract_schema(_happy_data(sensitivity="secret"))
    assert any("sensitivity" in e for e in errors)


def test_missing_http_urls_block_fails() -> None:
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    del ingest["http_urls"]
    errors = validate_ingest_contract_schema(data)
    assert any("http_urls" in e for e in errors)


def test_invalid_uuid_rejected() -> None:
    errors = validate_ingest_contract_schema(_happy_data(source_id="not-a-uuid"))
    assert any("UUID" in e for e in errors)


def test_unknown_fetch_key_rejected() -> None:
    # A bool typo on follow_redirects (or any new fetch knob) was previously
    # silent and fell through to defaults. The fetch-block validator catches
    # it now.
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["http_urls"]["fetch"] = {"timeoutSecs": 60}  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("timeoutSecs" in e for e in errors)


def test_follow_redirects_must_be_bool() -> None:
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["http_urls"]["fetch"] = {"follow_redirects": "yes"}  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("follow_redirects" in e and "boolean" in e for e in errors)


def test_http_urls_fetch_user_agent_accepted() -> None:
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["http_urls"]["fetch"] = {  # type: ignore[index]
        "retries": 3,
        "user_agent": "ExampleOrgDataPlatform/1.0 (contact: data@example.org)",
    }
    assert validate_ingest_contract_schema(data) == []


@pytest.mark.parametrize("bad", ["", "   ", 123, True, ["ua"]])
def test_http_urls_fetch_user_agent_must_be_non_empty_string(bad: object) -> None:
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["http_urls"]["fetch"] = {"user_agent": bad}  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("user_agent" in e and "non-empty string" in e for e in errors)


@pytest.mark.parametrize("bad", ["evil\r\nX-Injected: 1", "Org–DP/1.0", "tab\there"])
def test_http_urls_fetch_user_agent_must_be_printable_ascii(bad: str) -> None:
    # httpx encodes header values as strict ASCII, and a real transport
    # rejects CRLF only at request time (MockTransport never would) --
    # the loader is the reliable gate.
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["http_urls"]["fetch"] = {"user_agent": bad}  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("user_agent" in e and "printable ASCII" in e for e in errors)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("retries", "3"),
        ("retries", -1),
        ("retries", True),
        ("timeout_s", "fast"),
        ("timeout_s", 0),
        ("connect_timeout_s", -5),
    ],
)
def test_http_urls_fetch_numeric_knobs_validated(field: str, bad: object) -> None:
    # Before #413 a bad value loaded clean and crashed at materialize
    # time inside int()/float() coercion; now it fails at contract load.
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["http_urls"]["fetch"] = {field: bad}  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any(field in e for e in errors)


def test_http_urls_fetch_float_timeout_accepted() -> None:
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["http_urls"]["fetch"] = {"timeout_s": 0.5, "retries": 0}  # type: ignore[index]
    assert validate_ingest_contract_schema(data) == []


# ---------------------------------------------------------------------------
# validate_content (per #228)
# ---------------------------------------------------------------------------


def test_validate_content_omitted_is_fine() -> None:
    """The block is optional; omitting it preserves v0.27 union semantics."""
    data = _happy_data()
    errors = validate_ingest_contract_schema(data)
    assert errors == []


def test_validate_content_happy_path() -> None:
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["http_urls"]["validate_content"] = {  # type: ignore[index]
        "content_type_in": ["application/zip", "application/octet-stream"],
        "reject_first_bytes_match": ["<!DOCTYPE", "<html"],
        "max_first_bytes_check": 256,
    }
    errors = validate_ingest_contract_schema(data)
    assert errors == []


def test_validate_content_unknown_field_rejected() -> None:
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["http_urls"]["validate_content"] = {"mystery": "x"}  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("validate_content" in e and "mystery" in e for e in errors)


def test_validate_content_must_be_mapping() -> None:
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["http_urls"]["validate_content"] = ["not", "a", "mapping"]  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("validate_content" in e and "mapping" in e for e in errors)


def test_validate_content_empty_content_type_list_rejected() -> None:
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["http_urls"]["validate_content"] = {"content_type_in": []}  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("content_type_in" in e and "non-empty" in e for e in errors)


def test_validate_content_non_string_in_first_bytes_rejected() -> None:
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["http_urls"]["validate_content"] = {  # type: ignore[index]
        "reject_first_bytes_match": ["<html", 42],
    }
    errors = validate_ingest_contract_schema(data)
    assert any("reject_first_bytes_match" in e and "non-empty strings" in e for e in errors)


def test_validate_content_max_first_bytes_must_be_positive_int() -> None:
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["http_urls"]["validate_content"] = {"max_first_bytes_check": 0}  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("max_first_bytes_check" in e and "positive integer" in e for e in errors)


# ---------------------------------------------------------------------------
# extract_filter (ADR-1)
# ---------------------------------------------------------------------------


def test_extract_filter_default_is_empty_tuple(tmp_path: Path) -> None:
    """When omitted, ``extract_filter`` defaults to ``()`` -- the
    'no filter' sentinel that preserves Phase 1 'extract everything'
    behavior."""
    import yaml

    p = tmp_path / "cms_asp.ingest.yaml"
    p.write_text(yaml.safe_dump(_happy_data()))
    contract = load_ingest_contract(p)
    assert contract.extract_filter == ()


def test_extract_filter_happy_path(tmp_path: Path) -> None:
    import yaml

    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["extract_filter"] = ["meta/**", "subset/**"]

    p = tmp_path / "umls.ingest.yaml"
    p.write_text(yaml.safe_dump(data))
    contract = load_ingest_contract(p)
    assert contract.extract_filter == ("meta/**", "subset/**")


def test_extract_filter_empty_list_rejected() -> None:
    """ADR-1: empty list is a footgun (silently extracts nothing).
    Authors who mean 'extract everything' omit the field."""
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["extract_filter"] = []
    errors = validate_ingest_contract_schema(data)
    assert any("extract_filter" in e and "non-empty" in e for e in errors)


def test_extract_filter_must_be_list() -> None:
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["extract_filter"] = "meta/**"  # raw string, not a list
    errors = validate_ingest_contract_schema(data)
    assert any("extract_filter" in e and "list of strings" in e for e in errors)


def test_extract_filter_entries_must_be_non_empty_strings() -> None:
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["extract_filter"] = ["meta/**", ""]  # empty string disallowed
    errors = validate_ingest_contract_schema(data)
    assert any("extract_filter" in e and "non-empty" in e for e in errors)


def test_extract_filter_requires_extract_field() -> None:
    """The filter is meaningless without extraction; surface this at
    contract-load time so authors don't ship a filter that does nothing."""
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    del ingest["extract"]
    ingest["extract_filter"] = ["meta/**"]
    errors = validate_ingest_contract_schema(data)
    assert any("extract_filter" in e and "extract" in e for e in errors)


# ---------------------------------------------------------------------------
# payload_filename_template (#270)
# ---------------------------------------------------------------------------


def test_payload_filename_template_default_is_none(tmp_path: Path) -> None:
    """When omitted, ``payload_filename_template`` defaults to ``None``
    -- the precedence chain falls through to resolver hint /
    Content-Disposition / URL basename."""
    import yaml

    p = tmp_path / "demo.ingest.yaml"
    p.write_text(yaml.safe_dump(_happy_data()))
    contract = load_ingest_contract(p)
    assert contract.payload_filename_template is None


def test_payload_filename_template_round_trips(tmp_path: Path) -> None:
    """Authored template is parsed verbatim onto the dataclass; rendering
    happens at materialize time, not load time."""
    import yaml

    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["payload_filename_template"] = "{source_name}_{partition_key}.csv"

    p = tmp_path / "seer.ingest.yaml"
    p.write_text(yaml.safe_dump(data))
    contract = load_ingest_contract(p)
    assert contract.payload_filename_template == "{source_name}_{partition_key}.csv"


def test_payload_filename_template_must_be_string() -> None:
    """Type rejection at load time: a non-string template surfaces here
    rather than during materialization."""
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["payload_filename_template"] = 42
    errors = validate_ingest_contract_schema(data)
    assert any("payload_filename_template" in e for e in errors)


def test_payload_filename_template_empty_string_rejected() -> None:
    """An explicit empty string is a contract bug -- omit the field
    instead.  Symmetric with the ``extract_filter: []`` rejection."""
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["payload_filename_template"] = ""
    errors = validate_ingest_contract_schema(data)
    assert any("payload_filename_template" in e and "non-empty" in e for e in errors)


def test_payload_filename_template_unknown_field_typo_rejected() -> None:
    """Catches typos in the field name (the unknown-keys check still
    fires for the ingest inner block)."""
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["payload_filename_tempalte"] = "{source_name}.csv"  # typo
    errors = validate_ingest_contract_schema(data)
    assert any("payload_filename_tempalte" in e for e in errors)


# ---------------------------------------------------------------------------
# Multi-URL non-archive uniqueness check (#270)
# ---------------------------------------------------------------------------


def _non_archive_data() -> dict[str, Any]:
    """An http_urls contract shape with ``extract: []``."""
    return {
        "source_id": "11111111-1111-1111-1111-111111111111",
        "source_name": "demo",
        "sensitivity": "public",
        "ingest": {
            "pattern": "http_urls",
            "prefix": "demo/{partition_key}",
            "extract": [],  # non-archive
            "strip_extensions": [],
            "http_urls": {
                "fetch": {"retries": 0},
                "periods": [
                    {
                        "partition_key": "2024-01",
                        "urls": [],  # filled per-test
                    }
                ],
            },
        },
    }


def test_non_archive_multi_url_with_distinct_basenames_passes() -> None:
    """Two URLs whose sanitized basenames differ are accepted; each
    lands under its own descriptive name."""
    data = _non_archive_data()
    data["ingest"]["http_urls"]["periods"][0]["urls"] = [
        "https://example.com/file_a.csv",
        "https://example.com/file_b.csv",
    ]
    errors = validate_ingest_contract_schema(data)
    # No collision-related error; other unrelated errors must not exist
    # for this minimal happy-path contract either.
    assert not any("resolve to the same landed filename" in e for e in errors), errors


def test_non_archive_multi_url_with_colliding_basenames_rejected() -> None:
    """When two URLs share a sanitized basename, the loader raises so
    the silent collision can't ship to production."""
    data = _non_archive_data()
    data["ingest"]["http_urls"]["periods"][0]["urls"] = [
        "https://example.com/path-a/data.csv",
        "https://example.com/path-b/data.csv",
    ]
    errors = validate_ingest_contract_schema(data)
    assert any("data.csv" in e and "resolve to the same landed filename" in e for e in errors), (
        errors
    )


def test_non_archive_multi_url_with_template_collides_within_period() -> None:
    """Implication of bounded placeholders ({partition_key},
    {source_name}): a template renders identically for every URL in a
    single period.  Any multi-URL non-archive period that sets a
    template fails the uniqueness check."""
    data = _non_archive_data()
    data["ingest"]["payload_filename_template"] = "{source_name}_{partition_key}.csv"
    data["ingest"]["http_urls"]["periods"][0]["urls"] = [
        "https://example.com/file_a.csv",
        "https://example.com/file_b.csv",
    ]
    errors = validate_ingest_contract_schema(data)
    assert any("resolve to the same landed filename" in e for e in errors), errors


def test_archive_contract_with_multi_url_collisions_skipped() -> None:
    """Regression guard: archive contracts (``extract: ["zip"]``) are
    NOT subject to the uniqueness check; archive members keep their
    in-archive paths so the URL basename never reaches upload."""
    data = _non_archive_data()
    data["ingest"]["extract"] = ["zip"]  # archive contract
    data["ingest"]["http_urls"]["periods"][0]["urls"] = [
        "https://example.com/path-a/data.zip",
        "https://example.com/path-b/data.zip",  # would collide if non-archive
    ]
    errors = validate_ingest_contract_schema(data)
    assert not any("resolve to the same landed filename" in e for e in errors), errors


def test_non_archive_single_url_uniqueness_check_skipped() -> None:
    """Single-URL non-archive periods can't collide with themselves;
    the check should not fire."""
    data = _non_archive_data()
    data["ingest"]["http_urls"]["periods"][0]["urls"] = [
        "https://example.com/data.csv",
    ]
    errors = validate_ingest_contract_schema(data)
    assert not any("resolve to the same landed filename" in e for e in errors), errors


def test_loader_aggregates_multiple_validation_errors() -> None:
    """Audit-posture regression: the loader returns the full list of
    errors from a single contract load, not just the first one.  This
    test deliberately seeds three independent errors and asserts all
    three are reported."""
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["extract_filter"] = []  # error 1
    ingest["http_urls"]["fetch"] = {"timeoutSecs": 60}  # error 2 (unknown key)
    data["sensitivity"] = "secret"  # error 3 (bad enum)
    errors = validate_ingest_contract_schema(data)
    assert any("extract_filter" in e for e in errors)
    assert any("timeoutSecs" in e for e in errors)
    assert any("sensitivity" in e for e in errors)


# ---------------------------------------------------------------------------
# api_resolver block validation (ADR-2)
# ---------------------------------------------------------------------------


def _api_resolver_data(**ingest_overrides: object) -> dict[str, object]:
    """Happy-path ``api_resolver`` contract used by the validation tests."""
    ingest: dict[str, object] = {
        "pattern": "api_resolver",
        "prefix": "umls/{partition_key}",
        "extract": ["zip", "zip"],
        "extract_filter": ["meta/**"],
        "api_resolver": {
            "resolver": "uts_release",
            "resolver_config": {"release_type": "umls-full-release"},
            "credential": {"secret_name": "uts-api-key"},
            "partition": {"mode": "dynamic", "key_from": "release_version"},
            "idempotency": "hash_compare",
        },
    }
    ingest.update(ingest_overrides)
    return {
        "source_id": _UUID,
        "source_name": "umls-meta",
        "sensitivity": "confidential",
        "data_owner": "data-platform",
        "compliance_review": "SECURITY.md#umls",
        "ingest": ingest,
    }


def test_api_resolver_happy_path() -> None:
    errors = validate_ingest_contract_schema(_api_resolver_data())
    assert errors == []


def test_api_resolver_fetch_accepted_with_user_agent() -> None:
    data = _api_resolver_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["api_resolver"]["fetch"] = {  # type: ignore[index]
        "retries": 2,
        "timeout_s": 3600,
        "connect_timeout_s": 30,
        "user_agent": "ExampleOrgDataPlatform/1.0 (contact: data@example.org)",
    }
    assert validate_ingest_contract_schema(data) == []


def test_api_resolver_fetch_unknown_key_rejected() -> None:
    # Before #413 the api_resolver fetch block's contents were unchecked,
    # so a typo'd knob silently fell through to defaults.
    data = _api_resolver_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["api_resolver"]["fetch"] = {"timeoutSecs": 60}  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.api_resolver.fetch" in e and "timeoutSecs" in e for e in errors)


def test_api_resolver_fetch_follow_redirects_rejected() -> None:
    # The resolved-URL download always follows redirects; the knob is
    # http_urls-only, so it is rejected rather than silently ignored.
    data = _api_resolver_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["api_resolver"]["fetch"] = {"follow_redirects": False}  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("follow_redirects" in e and "unknown" in e for e in errors)


def test_api_resolver_fetch_must_be_mapping() -> None:
    data = _api_resolver_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["api_resolver"]["fetch"] = "fast"  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.api_resolver.fetch" in e and "mapping" in e for e in errors)


def test_api_resolver_fetch_user_agent_must_be_non_empty_string() -> None:
    data = _api_resolver_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["api_resolver"]["fetch"] = {"user_agent": ""}  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.api_resolver.fetch.user_agent" in e for e in errors)


def test_api_resolver_fetch_numeric_knobs_validated() -> None:
    # The shared fetch validator's value-type checks apply to
    # api_resolver too, not just http_urls.
    data = _api_resolver_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["api_resolver"]["fetch"] = {"timeout_s": "fast"}  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.api_resolver.fetch.timeout_s" in e for e in errors)


def test_api_resolver_missing_inner_block_rejected() -> None:
    data = _api_resolver_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    del ingest["api_resolver"]
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.api_resolver" in e and "required" in e for e in errors)


def test_api_resolver_unknown_resolver_name_rejected() -> None:
    data = _api_resolver_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["api_resolver"]["resolver"] = "mystery_release"  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("Unknown release resolver 'mystery_release'" in e for e in errors)


def test_api_resolver_unknown_resolver_config_field_rejected() -> None:
    """ADR-2: per-resolver validate_config dispatched at contract-load
    time and unknown keys are flagged so a typo like
    ``releas_type: ...`` fails at deploy."""
    data = _api_resolver_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["api_resolver"]["resolver_config"]["releas_type"] = "typo"  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any(
        "ingest.api_resolver.resolver_config.releas_type" in e and "unknown" in e for e in errors
    )


def test_api_resolver_resolver_config_must_be_mapping() -> None:
    data = _api_resolver_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["api_resolver"]["resolver_config"] = ["not", "a", "mapping"]  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.api_resolver.resolver_config" in e and "mapping" in e for e in errors)


def test_api_resolver_credential_secret_name_required() -> None:
    data = _api_resolver_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    del ingest["api_resolver"]["credential"]["secret_name"]  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("credential.secret_name" in e and "required" in e for e in errors)


def test_api_resolver_without_credential_block_validates() -> None:
    """Per #218: ``credential`` is optional; resolvers that don't
    authenticate (e.g. ``calendar``) omit the block entirely."""
    data = _api_resolver_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    del ingest["api_resolver"]["credential"]  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert errors == []


def test_api_resolver_credential_block_invalid_type_rejected() -> None:
    """When the credential block IS present, structural validation still applies."""
    data = _api_resolver_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["api_resolver"]["credential"] = "not-a-mapping"  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.api_resolver.credential" in e and "mapping" in e for e in errors)


def test_api_resolver_partition_mode_must_be_dynamic() -> None:
    data = _api_resolver_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["api_resolver"]["partition"]["mode"] = "static"  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.api_resolver.partition.mode" in e and "must be one of" in e for e in errors)


def test_api_resolver_partition_key_from_required() -> None:
    data = _api_resolver_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    del ingest["api_resolver"]["partition"]["key_from"]  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("partition.key_from" in e and "required" in e for e in errors)


def test_api_resolver_block_unknown_field_rejected() -> None:
    data = _api_resolver_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["api_resolver"]["unknown_top_level"] = "x"  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.api_resolver" in e and "unknown_top_level" in e for e in errors)


def test_api_resolver_aggregates_per_resolver_errors() -> None:
    """Audit-posture: per-resolver validate_config returns all errors,
    not just the first.  Combine resolver-level errors with structural
    errors and assert all three categories surface."""
    data = _api_resolver_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    # Resolver-level: drop the required release_type
    del ingest["api_resolver"]["resolver_config"]["release_type"]  # type: ignore[index]
    # Resolver-level: add an unknown key
    ingest["api_resolver"]["resolver_config"]["foo"] = 1  # type: ignore[index]
    # Structural: drop partition (still required even though credential is optional)
    del ingest["api_resolver"]["partition"]  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert any("release_type" in e and "required" in e for e in errors)
    assert any("foo" in e and "unknown" in e for e in errors)
    assert any("partition" in e and "required" in e for e in errors)


# ---------------------------------------------------------------------------
# api_crawl block validation (#415)
# ---------------------------------------------------------------------------


@pytest.fixture
def _stub_crawl_plan() -> Iterator[type]:
    """Register a minimal crawl plan for load-time lookup; restore after."""
    from moncpipelib.ingest.crawl_plans import CRAWL_PLANS, register_crawl_plan

    class _RxStubPlan:
        name = "rxnav_stub"

        def validate_config(self, config: dict[str, Any]) -> list[str]:
            errors: list[str] = []
            unknown = set(config) - {"rela_sources"}
            errors.extend(f"unknown field {k!r}" for k in sorted(unknown))
            if "rela_sources" in config and not isinstance(config["rela_sources"], list):
                errors.append("rela_sources: must be a list")
            return errors

        def crawl(self, client: Any, api_key: Any, config: Any, ctx: Any) -> Any:
            raise NotImplementedError("load-time stub; never crawled")

    before = dict(CRAWL_PLANS)
    register_crawl_plan(_RxStubPlan())
    try:
        yield _RxStubPlan
    finally:
        CRAWL_PLANS.clear()
        CRAWL_PLANS.update(before)


def _api_crawl_data(**ingest_overrides: object) -> dict[str, object]:
    """Happy-path ``api_crawl`` contract used by the validation tests."""
    ingest: dict[str, object] = {
        "pattern": "api_crawl",
        "prefix": "rxclass/{partition_key}",
        "api_crawl": {
            "crawl_plan": "rxnav_stub",
            "crawl_config": {"rela_sources": ["ATC", "MESH"]},
            "resolver": "calendar",
            "resolver_config": {
                "start_date": "2026-06-01",
                "cadence": "monthly",
                "url": "https://rxnav.nlm.nih.gov/REST/rxclass/allClasses.json",
            },
            "partition": {"mode": "dynamic", "key_from": "partition_key"},
            "rate_limit_rps": 5,
        },
    }
    ingest.update(ingest_overrides)
    return {
        "source_id": _UUID,
        "source_name": "rxclass",
        "sensitivity": "public",
        "data_owner": "data-platform",
        "ingest": ingest,
    }


def _api_crawl_block(data: dict[str, object]) -> dict[str, Any]:
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    block = ingest["api_crawl"]
    assert isinstance(block, dict)
    return block


def test_api_crawl_happy_path(_stub_crawl_plan: type) -> None:
    assert validate_ingest_contract_schema(_api_crawl_data()) == []


def test_api_crawl_pattern_config_round_trips(_stub_crawl_plan: type, tmp_path: Path) -> None:
    import yaml

    p = tmp_path / "rxclass.ingest.yaml"
    p.write_text(yaml.safe_dump(_api_crawl_data()))
    contract = load_ingest_contract(p)
    assert contract.pattern == "api_crawl"
    assert contract.pattern_config["crawl_plan"] == "rxnav_stub"
    assert contract.pattern_config["rate_limit_rps"] == 5
    assert contract.pattern_config["resolver"] == "calendar"


def test_api_crawl_missing_inner_block_rejected() -> None:
    data = _api_crawl_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    del ingest["api_crawl"]
    errors = validate_ingest_contract_schema(data)
    assert any("'ingest.api_crawl' is required" in e for e in errors)


def test_api_crawl_missing_crawl_plan_rejected(_stub_crawl_plan: type) -> None:
    data = _api_crawl_data()
    del _api_crawl_block(data)["crawl_plan"]
    errors = validate_ingest_contract_schema(data)
    assert any("'ingest.api_crawl.crawl_plan' is required" in e for e in errors)


def test_api_crawl_unknown_crawl_plan_lists_known(_stub_crawl_plan: type) -> None:
    data = _api_crawl_data()
    _api_crawl_block(data)["crawl_plan"] = "nope"
    errors = validate_ingest_contract_schema(data)
    assert any("Unknown crawl plan 'nope'" in e and "rxnav_stub" in e for e in errors)


def test_api_crawl_crawl_config_dispatched_to_plan(_stub_crawl_plan: type) -> None:
    data = _api_crawl_data()
    _api_crawl_block(data)["crawl_config"] = {"rela_surces": ["ATC"]}  # typo
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.api_crawl.crawl_config" in e and "rela_surces" in e for e in errors)


def test_api_crawl_missing_rate_limit_carries_guidance(_stub_crawl_plan: type) -> None:
    """Per maintainer review on the #415 plan: the missing-field error
    explains the requests-per-second budget and the upstream-cap
    rationale, not just field presence."""
    data = _api_crawl_data()
    del _api_crawl_block(data)["rate_limit_rps"]
    errors = validate_ingest_contract_schema(data)
    (msg,) = [e for e in errors if "rate_limit_rps" in e]
    assert "requests-per-second budget" in msg
    assert "upstream's published cap" in msg
    assert "20 req/s" in msg


@pytest.mark.parametrize("bad_rps", [0, -1, "5", True, None])
def test_api_crawl_non_positive_or_non_numeric_rate_limit_rejected(
    _stub_crawl_plan: type, bad_rps: object
) -> None:
    data = _api_crawl_data()
    _api_crawl_block(data)["rate_limit_rps"] = bad_rps
    errors = validate_ingest_contract_schema(data)
    assert any("'ingest.api_crawl.rate_limit_rps' must be a number > 0" in e for e in errors)


def test_api_crawl_unknown_keys_rejected(_stub_crawl_plan: type) -> None:
    data = _api_crawl_data()
    _api_crawl_block(data)["rate_limit"] = 5  # typo'd field name
    errors = validate_ingest_contract_schema(data)
    assert any("rate_limit" in e and "unknown" in e for e in errors)


def test_api_crawl_fetch_follow_redirects_rejected(_stub_crawl_plan: type) -> None:
    # Crawl GETs hardcode follow_redirects=True; the knob is
    # http_urls-only, so it is rejected rather than silently ignored.
    data = _api_crawl_data()
    _api_crawl_block(data)["fetch"] = {"follow_redirects": False}
    errors = validate_ingest_contract_schema(data)
    assert any("follow_redirects" in e and "unknown" in e for e in errors)


def test_api_crawl_resolver_credential_partition_validated(_stub_crawl_plan: type) -> None:
    """The shared resolver-backed sub-blocks get the same validation as
    api_resolver, with api_crawl-prefixed messages."""
    data = _api_crawl_data()
    block = _api_crawl_block(data)
    block["resolver_config"] = {"cadence": "monthly"}  # missing start_date + url
    block["credential"] = {"secretname": "typo"}
    del block["partition"]
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.api_crawl.resolver_config" in e and "start_date" in e for e in errors)
    assert any("ingest.api_crawl.credential" in e and "secretname" in e for e in errors)
    assert any("'ingest.api_crawl.partition' is required" in e for e in errors)


def test_api_crawl_credential_optional(_stub_crawl_plan: type) -> None:
    """Public APIs (RxClass) omit the credential block entirely."""
    data = _api_crawl_data()
    assert "credential" not in _api_crawl_block(data)
    assert validate_ingest_contract_schema(data) == []
