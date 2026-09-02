"""Tests for load_ingest_contract + validate_ingest_contract_schema."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
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


def test_sensitivity_mapping_value_rejected_not_raised() -> None:
    """A mapping value is unhashable, and a bare membership test against
    ``KNOWN_SENSITIVITIES`` would raise ``TypeError`` instead of
    returning a clean error. Must produce exactly one error naming the
    valid set instead of crashing the loader."""
    errors = validate_ingest_contract_schema(_happy_data(sensitivity={"a": "b"}))
    sensitivity_errors = [e for e in errors if "sensitivity" in e]
    assert len(sensitivity_errors) == 1
    assert "must be one of" in sensitivity_errors[0]


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


# ---------------------------------------------------------------------------
# ingest.pattern validation against the live registry (#464)
# ---------------------------------------------------------------------------


def test_unknown_ingest_pattern_rejected_and_lists_known() -> None:
    """The issue's headline case: an unregistered pattern name (e.g. the
    rejected ``manual_drop`` non-goal from #463) must fail at contract
    load, not surface first as a ``get_pattern`` ``KeyError`` at
    materialize time."""
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["pattern"] = "manual_drop"
    errors = validate_ingest_contract_schema(data)
    (msg,) = [e for e in errors if "manual_drop" in e]
    assert "http_urls" in msg
    assert "blob_mirror" in msg


def test_unknown_ingest_pattern_error_shape_matches_get_pattern() -> None:
    """The loader delegates to get_pattern's own KeyError message instead
    of re-deriving it (#464/#467), so the two are no longer
    byte-identical -- the loader message additionally carries the
    'ingest.pattern' field path every other loader error carries. Assert
    the loader message CONTAINS get_pattern's raw message verbatim and
    is prefixed with the field path -- a stronger pin than the previous
    substring pair, since it still guarantees the two read consistently."""
    from moncpipelib.ingest.patterns import get_pattern

    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["pattern"] = "manual_drop"
    errors = validate_ingest_contract_schema(data)
    (msg,) = [e for e in errors if "manual_drop" in e]

    with pytest.raises(KeyError) as exc_info:
        get_pattern("manual_drop")
    # .args[0] is the raw message string; str(exc_info.value) would add
    # KeyError's outer repr quotes and break the containment check.
    assert exc_info.value.args[0] in msg
    assert msg.startswith("'ingest.pattern':")


class _StubValidationPattern:
    """Minimal registered pattern with no dedicated block validator."""

    def __init__(self, name: str) -> None:
        self.name = name

    def discover_partitions(self, contract: Any, ctx: Any) -> list[Any]:
        del contract, ctx
        return []

    def materialize_partition(
        self, contract: Any, partition_spec: Any, blob: Any, ctx: Any
    ) -> list[Any]:
        del contract, partition_spec, blob, ctx
        return []


@contextmanager
def _registered_stub_pattern(name: str) -> Iterator[None]:
    """Register a no-op pattern under ``name`` for the block, then restore.

    ``INGEST_PATTERNS`` is module-global and ``validate_ingest_contract_schema``
    reads it live, so every test that exercises the consumer-registration
    seam (#399) must leave the registry exactly as it found it -- including
    restoring a pre-existing entry rather than deleting it.
    """
    from moncpipelib.ingest.patterns import INGEST_PATTERNS, register_pattern

    original = INGEST_PATTERNS.get(name)
    register_pattern(_StubValidationPattern(name))  # type: ignore[arg-type]
    try:
        yield
    finally:
        if original is not None:
            register_pattern(original)
        else:
            INGEST_PATTERNS.pop(name, None)


def test_registered_custom_pattern_and_block_accepted() -> None:
    """#399 extension seam: a consumer-registered pattern (and its
    identically-named inner block) must be accepted by the generic
    loader once imported/registered, without needing a builtin case in
    ``KNOWN_INGEST_INNER_FIELDS``."""
    with _registered_stub_pattern("_test_validation"):
        data = _happy_data()
        ingest = data["ingest"]
        assert isinstance(ingest, dict)
        del ingest["http_urls"]
        ingest["pattern"] = "_test_validation"
        ingest["_test_validation"] = {}
        errors = validate_ingest_contract_schema(data)
        assert errors == []


def test_unregistered_pattern_block_key_still_rejected() -> None:
    """Regression guard: unioning the live registry into the known-keys
    set must not turn the inner ``ingest`` block into a free-for-all --
    a key that names no registered pattern is still an unknown field."""
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["not_a_pattern"] = {}
    errors = validate_ingest_contract_schema(data)
    assert any("not_a_pattern" in e and "unknown field" in e for e in errors)


def test_pattern_block_validator_map_covers_all_builtins() -> None:
    """Pins ``_PATTERN_BLOCK_VALIDATORS`` against the single source of
    truth for the builtin pattern set: ``BUILTIN_PATTERN_NAMES``,
    snapshotted from ``INGEST_PATTERNS`` at import time in
    ``patterns/__init__.py``, immediately after ``_register_builtin_patterns()``
    runs and before any consumer registration can widen the registry.

    The map replaced a hand-written if/elif dispatch chain specifically
    so a forgotten entry cannot silently skip block validation for a new
    builtin pattern (the next one due is ``browser_export``, #463). A
    hardcoded literal set on this side previously left that gap open:
    registering a 5th builtin pattern without a map entry left both sides
    at the same four-name literal, so the test stayed green (#467).
    Comparing against the immutable snapshot instead of the
    live ``INGEST_PATTERNS`` registry keeps this immune to another
    test's leaked registration while still catching a genuinely
    forgotten builtin entry, since the snapshot only grows when
    ``_register_builtin_patterns()`` itself grows.
    """
    from moncpipelib.contracts.loader import _PATTERN_BLOCK_VALIDATORS
    from moncpipelib.ingest.patterns import BUILTIN_PATTERN_NAMES

    assert set(_PATTERN_BLOCK_VALIDATORS) == BUILTIN_PATTERN_NAMES


# ---------------------------------------------------------------------------
# ingest.<pattern>.idempotency mode validation (#464)
# ---------------------------------------------------------------------------


def test_idempotency_unknown_mode_rejected_http_urls() -> None:
    """The realistic typo (`hash_compre`) previously passed validation
    silently; it must now be caught at contract load."""
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["http_urls"]["idempotency"] = "hash_compre"  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    (msg,) = [e for e in errors if "idempotency" in e]
    assert "hash_compare" in msg


def test_idempotency_unknown_mode_rejected_api_resolver() -> None:
    data = _api_resolver_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["api_resolver"]["idempotency"] = "hash_compre"  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    (msg,) = [e for e in errors if "idempotency" in e]
    assert "hash_compare" in msg


def test_idempotency_hash_compare_still_accepted() -> None:
    """Non-breaking regression guard: several data-platform contracts set
    ``idempotency: hash_compare`` today (see the downstream audit in
    ``docs/migrations/20260805_464-contract-validation-hardening.md``); the
    value must remain valid."""
    assert validate_ingest_contract_schema(_happy_data()) == []


@pytest.mark.parametrize("bad", [True, 123, {"a": "b"}, ["a"]])
def test_idempotency_non_string_rejected(bad: object) -> None:
    """A non-string value must produce exactly one ``idempotency``
    error, not a duplicate pair from separate type/enum checks. Includes
    unhashable values (dict, list) -- legal YAML a contract author can
    write -- which must fail clean rather than raise ``TypeError`` from
    a bare membership test against the frozenset of known modes."""
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["http_urls"]["idempotency"] = bad  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    idempotency_errors = [e for e in errors if "idempotency" in e]
    assert len(idempotency_errors) == 1


# ---------------------------------------------------------------------------
# Shared _validate_enum_value routing (#464/#467)
# ---------------------------------------------------------------------------


def _bad_http_urls_idempotency(bad: object) -> list[str]:
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["http_urls"]["idempotency"] = bad  # type: ignore[index]
    return validate_ingest_contract_schema(data)


def _bad_api_resolver_idempotency(bad: object) -> list[str]:
    data = _api_resolver_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["api_resolver"]["idempotency"] = bad  # type: ignore[index]
    return validate_ingest_contract_schema(data)


def _bad_sensitivity(bad: object) -> list[str]:
    return validate_ingest_contract_schema(_happy_data(sensitivity=bad))


def _bad_partition_mode(bad: object) -> list[str]:
    data = _api_resolver_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["api_resolver"]["partition"]["mode"] = bad  # type: ignore[index]
    return validate_ingest_contract_schema(data)


@pytest.mark.parametrize("bad", [{"a": "b"}, ["a"]], ids=["mapping", "list"])
@pytest.mark.parametrize(
    "field_marker,build_errors",
    [
        ("idempotency", _bad_http_urls_idempotency),
        ("idempotency", _bad_api_resolver_idempotency),
        ("sensitivity", _bad_sensitivity),
        ("partition.mode", _bad_partition_mode),
    ],
    ids=["http_urls.idempotency", "api_resolver.idempotency", "sensitivity", "partition.mode"],
)
def test_enum_field_unhashable_value_rejected_cleanly(
    field_marker: str, build_errors: Callable[[object], list[str]], bad: object
) -> None:
    """Every field routed through the shared ``_validate_enum_value``
    helper in ``validate_ingest_contract_schema`` must produce exactly
    one error and never raise for a mapping or list value -- a bare
    ``value in known_set`` membership test calls ``hash(value)`` and
    crashes on these otherwise-legal YAML values (#464/#467)."""
    errors = build_errors(bad)
    matches = [e for e in errors if field_marker in e]
    assert len(matches) == 1
    assert "must be one of" in matches[0]


# ---------------------------------------------------------------------------
# Registered custom pattern block must be a mapping (#467)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_block", ["not-a-mapping", None], ids=["string", "null"])
def test_registered_custom_pattern_non_mapping_block_rejected_not_raised(
    bad_block: object, tmp_path: Path
) -> None:
    """Regression guard for a defect this PR's registry union introduced
    ``load_ingest_contract`` does
    ``dict(ingest.get(pattern, {}))``. On ``main`` the unknown-field
    sweep accidentally caught a non-mapping block for an unregistered
    pattern name; unioning the live registry into the known-keys set
    removed that accidental protection for a *registered* custom pattern
    with no dedicated block validator, so a string or null block crashed
    ``load_ingest_contract`` with ``ValueError``/``TypeError`` instead of
    raising ``ContractValidationError`` cleanly. Covers both a string
    block and an explicit YAML null block (``pattern in ingest`` but the
    value isn't a mapping)."""
    import yaml

    pattern_name = "_test_validation_block_shape"
    with _registered_stub_pattern(pattern_name):
        data = _happy_data()
        ingest = data["ingest"]
        assert isinstance(ingest, dict)
        del ingest["http_urls"]
        ingest["pattern"] = pattern_name
        ingest[pattern_name] = bad_block
        errors = validate_ingest_contract_schema(data)
        assert any(f"'ingest.{pattern_name}' must be a mapping" in e for e in errors)

        p = tmp_path / "x.ingest.yaml"
        p.write_text(yaml.safe_dump(data))
        with pytest.raises(ContractValidationError):
            load_ingest_contract(p)


def test_registered_custom_pattern_absent_block_still_fine() -> None:
    """Regression guard: the mapping requirement must not turn into a
    'block is required' requirement -- ``dict(ingest.get(pattern, {}))``
    already handles a wholly absent block, per #399's original design."""
    pattern_name = "_test_validation_absent_block"
    with _registered_stub_pattern(pattern_name):
        data = _happy_data()
        ingest = data["ingest"]
        assert isinstance(ingest, dict)
        del ingest["http_urls"]
        ingest["pattern"] = pattern_name
        errors = validate_ingest_contract_schema(data)
        assert errors == []


# ---------------------------------------------------------------------------
# Suppress the misleading duplicate "unknown field" error for an
# unregistered pattern's own block key (#467)
# ---------------------------------------------------------------------------


def test_unregistered_pattern_own_block_key_not_double_reported() -> None:
    """``manual_drop`` is #463's rejected non-goal: an unregistered
    pattern name whose block key is real config for the (unregistered)
    pattern it names, not an illegal key. Before this fix,
    ``ingest: {pattern: manual_drop, manual_drop: {...}}`` produced both
    an ``unknown field 'manual_drop'`` error (false in substance) and
    the ``Unknown ingest pattern 'manual_drop'`` error -- the first
    misdirects an author into deleting the block instead of fixing
    ``pattern:``. Exactly one ``manual_drop`` error must survive, and it
    must be the unknown-pattern one."""
    data = _happy_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    del ingest["http_urls"]
    ingest["pattern"] = "manual_drop"
    ingest["manual_drop"] = {"dir": "/drop"}
    errors = validate_ingest_contract_schema(data)
    matches = [e for e in errors if "manual_drop" in e]
    (msg,) = matches
    assert "Unknown ingest pattern" in msg
    assert "unknown field" not in msg


# ---------------------------------------------------------------------------
# browser_export block validation (#463)
# ---------------------------------------------------------------------------


@pytest.fixture
def _stub_export_plan() -> Iterator[type]:
    """Register a single-file + a multi-file export plan; restore after."""
    from moncpipelib.ingest.export_plans import EXPORT_PLANS, register_export_plan

    class _StubExportPlan:
        name = "_stub_export_plan"

        def validate_config(self, config: dict[str, Any]) -> list[str]:
            unknown = set(config) - {"report_id"}
            return [f"unknown field {k!r}" for k in sorted(unknown)]

        def export(self, session: Any, partition_key: Any, config: Any, ctx: Any) -> Any:
            raise NotImplementedError("load-time stub; never exported")

    class _StubMultiFileExportPlan:
        name = "_stub_multi_file_export_plan"
        multi_file = True

        def validate_config(self, config: dict[str, Any]) -> list[str]:
            del config
            return []

        def export(self, session: Any, partition_key: Any, config: Any, ctx: Any) -> Any:
            raise NotImplementedError("load-time stub; never exported")

    before = dict(EXPORT_PLANS)
    register_export_plan(_StubExportPlan())
    register_export_plan(_StubMultiFileExportPlan())
    try:
        yield _StubExportPlan
    finally:
        EXPORT_PLANS.clear()
        EXPORT_PLANS.update(before)


def _browser_export_data(**data_overrides: object) -> dict[str, object]:
    """Happy-path ``browser_export`` contract used by the validation tests."""
    ingest: dict[str, object] = {
        "pattern": "browser_export",
        "prefix": "340b/{partition_key}",
        "browser_export": {
            "export_plan": "_stub_export_plan",
            "export_config": {},
            "allowed_hosts": ["340bopais.hrsa.gov"],
            "partition": {
                "mode": "dynamic",
                "cadence": "daily",
                "anchor_tz": "America/New_York",
            },
            "fetch": {"user_agent": "example.com/340b-ingest (contact: data-platform)"},
        },
    }
    data: dict[str, object] = {
        "source_id": _UUID,
        "source_name": "340b-ceiling-price",
        "sensitivity": "public",
        "compliance_review": "SECURITY.md#340b",
        "ingest": ingest,
    }
    data.update(data_overrides)
    return data


def _browser_export_block(data: dict[str, object]) -> dict[str, Any]:
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    block = ingest["browser_export"]
    assert isinstance(block, dict)
    return block


def test_browser_export_happy_contract_validates_clean(_stub_export_plan: type) -> None:
    assert validate_ingest_contract_schema(_browser_export_data()) == []


def test_browser_export_block_required(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    del ingest["browser_export"]
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.browser_export" in e and "required" in e for e in errors)


def test_browser_export_block_must_be_mapping(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    ingest = data["ingest"]
    assert isinstance(ingest, dict)
    ingest["browser_export"] = "not-a-mapping"
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.browser_export" in e and "mapping" in e for e in errors)


def test_browser_export_unknown_field_rejected(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["mystery"] = "x"
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.browser_export" in e and "mystery" in e for e in errors)


def test_export_plan_required(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    del _browser_export_block(data)["export_plan"]
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.browser_export.export_plan" in e and "required" in e for e in errors)


def test_export_plan_unknown_names_known_plans(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["export_plan"] = "mystery_plan"
    errors = validate_ingest_contract_schema(data)
    (msg,) = [e for e in errors if "export_plan" in e]
    assert "Unknown export plan" in msg
    assert "_stub_export_plan" in msg


def test_export_config_unknown_key_rejected_via_plan_validate_config(
    _stub_export_plan: type,
) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["export_config"] = {"typo_field": 1}
    errors = validate_ingest_contract_schema(data)
    assert any(e.startswith("'ingest.browser_export.export_config.") for e in errors)


def test_allowed_hosts_required(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    del _browser_export_block(data)["allowed_hosts"]
    errors = validate_ingest_contract_schema(data)
    assert any("allowed_hosts" in e and "required" in e for e in errors)


def test_allowed_hosts_empty_rejected(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["allowed_hosts"] = []
    errors = validate_ingest_contract_schema(data)
    assert any("allowed_hosts" in e and "non-empty" in e for e in errors)


def test_allowed_hosts_entry_with_scheme_rejected(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["allowed_hosts"] = ["https://x.gov"]
    errors = validate_ingest_contract_schema(data)
    assert any("allowed_hosts[0]" in e for e in errors)


def test_allowed_hosts_entry_with_port_rejected(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["allowed_hosts"] = ["x.gov:443"]
    errors = validate_ingest_contract_schema(data)
    assert any("allowed_hosts[0]" in e for e in errors)


def test_allowed_hosts_entry_with_wildcard_rejected(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["allowed_hosts"] = ["*.hrsa.gov"]
    errors = validate_ingest_contract_schema(data)
    assert any("allowed_hosts[0]" in e for e in errors)


def test_allowed_hosts_entry_with_path_rejected(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["allowed_hosts"] = ["x.gov/reports"]
    errors = validate_ingest_contract_schema(data)
    assert any("allowed_hosts[0]" in e for e in errors)


@pytest.mark.parametrize(
    "entry",
    ["localhost", "169.254.169.254", "kubernetes.default.svc", "printer.local"],
)
def test_allowed_hosts_entry_shape_only_accepts_previously_denied_names(
    _stub_export_plan: type, entry: str
) -> None:
    """Round 6 (docs/migrations/20260807_463-egress-allowlist-reframe.md):
    the deny-list is removed by deliberate decision -- ``allowed_hosts``
    validation is shape-only. An entry naming loopback, an IP literal, or
    an in-cluster service now produces NO ``allowed_hosts`` error, pinned
    deliberately so deny logic cannot silently re-accrete."""
    data = _browser_export_data()
    _browser_export_block(data)["allowed_hosts"] = [entry]
    errors = validate_ingest_contract_schema(data)
    assert not any("allowed_hosts" in e for e in errors), errors


def test_allowed_hosts_entry_ordinary_public_hostname_accepted(_stub_export_plan: type) -> None:
    """No over-rejection: an ordinary public hostname -- including one with
    an ``internal`` label that is NOT its last label -- must validate
    cleanly under the shape-only screen."""
    data = _browser_export_data()
    _browser_export_block(data)["allowed_hosts"] = ["340bopais.hrsa.gov", "internal.example.com"]
    assert validate_ingest_contract_schema(data) == []


def test_allowed_hosts_entry_ordinary_hostname_with_trailing_dot_accepted(
    _stub_export_plan: type,
) -> None:
    """The DNS-root-label form of an ordinary public hostname is not
    itself a reserved name and must still validate cleanly -- the fix
    strips the trailing dot only for the purposes of the reserved-name
    comparison, it does not reject every trailing-dot hostname outright."""
    data = _browser_export_data()
    _browser_export_block(data)["allowed_hosts"] = ["340bopais.hrsa.gov."]
    assert validate_ingest_contract_schema(data) == []


def test_partition_required(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    del _browser_export_block(data)["partition"]
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.browser_export.partition" in e and "required" in e for e in errors)


def test_partition_mode_must_be_dynamic(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["partition"]["mode"] = "static"
    errors = validate_ingest_contract_schema(data)
    assert any(
        "ingest.browser_export.partition.mode" in e and "must be one of" in e for e in errors
    )


def test_partition_mode_required(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    del _browser_export_block(data)["partition"]["mode"]
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.browser_export.partition.mode" in e and "required" in e for e in errors)


def test_partition_cadence_required(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    del _browser_export_block(data)["partition"]["cadence"]
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.browser_export.partition.cadence" in e and "required" in e for e in errors)


def test_partition_cadence_unknown_rejected(_stub_export_plan: type) -> None:
    from moncpipelib.contracts.loader import KNOWN_BROWSER_EXPORT_CADENCES

    data = _browser_export_data()
    _browser_export_block(data)["partition"]["cadence"] = "yearly"
    errors = validate_ingest_contract_schema(data)
    (msg,) = [
        e
        for e in errors
        if "ingest.browser_export.partition.cadence" in e and "must be one of" in e
    ]
    assert str(sorted(KNOWN_BROWSER_EXPORT_CADENCES)) in msg


def test_partition_cadence_weekly_rejected_with_reason(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["partition"]["cadence"] = "weekly"
    errors = validate_ingest_contract_schema(data)
    reason_matches = [e for e in errors if "anchor_dow" in e]
    assert len(reason_matches) == 1


def test_anchor_tz_required(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    del _browser_export_block(data)["partition"]["anchor_tz"]
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.browser_export.partition.anchor_tz" in e and "required" in e for e in errors)


def test_anchor_tz_unknown_timezone_rejected(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["partition"]["anchor_tz"] = "Mars/Olympus"
    errors = validate_ingest_contract_schema(data)
    assert any(
        "ingest.browser_export.partition.anchor_tz" in e and "unknown timezone" in e for e in errors
    )


def test_anchor_tz_malformed_value_does_not_raise(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["partition"]["anchor_tz"] = "../../etc/passwd"
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.browser_export.partition.anchor_tz" in e for e in errors)


def test_partition_block_mapping_value_does_not_crash_validator(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    block = _browser_export_block(data)
    block["partition"]["cadence"] = {"a": "b"}
    block["partition"]["mode"] = [1, 2]
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.browser_export.partition.cadence" in e for e in errors)
    assert any("ingest.browser_export.partition.mode" in e for e in errors)


def test_publication_lag_hours_negative_rejected(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["partition"]["publication_lag_hours"] = -1
    errors = validate_ingest_contract_schema(data)
    assert any("publication_lag_hours" in e and "non-negative" in e for e in errors)


def test_publication_lag_hours_bool_rejected(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["partition"]["publication_lag_hours"] = True
    errors = validate_ingest_contract_schema(data)
    assert any("publication_lag_hours" in e and "non-negative" in e for e in errors)


def test_browser_block_unknown_field_rejected(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["browser"] = {"mystery": "x"}
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.browser_export.browser" in e and "mystery" in e for e in errors)


def test_browser_headless_must_be_bool(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["browser"] = {"headless": "yes"}
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.browser_export.browser.headless" in e and "boolean" in e for e in errors)


def test_browser_timeouts_must_be_positive_numbers(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["browser"] = {
        "navigation_timeout_s": 0,
        "download_timeout_s": -5,
    }
    errors = validate_ingest_contract_schema(data)
    assert any("navigation_timeout_s" in e and "positive" in e for e in errors)
    assert any("download_timeout_s" in e and "positive" in e for e in errors)


def test_fetch_retries_rejected_as_unknown_field(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["fetch"] = {"retries": 3}
    errors = validate_ingest_contract_schema(data)
    assert any(
        "ingest.browser_export.fetch" in e and "retries" in e and "unknown" in e for e in errors
    )


def test_fetch_timeout_s_rejected_as_unknown_field(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["fetch"] = {"timeout_s": 30}
    errors = validate_ingest_contract_schema(data)
    assert any(
        "ingest.browser_export.fetch" in e and "timeout_s" in e and "unknown" in e for e in errors
    )


def test_fetch_connect_timeout_s_rejected_as_unknown_field(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["fetch"] = {"connect_timeout_s": 5}
    errors = validate_ingest_contract_schema(data)
    assert any(
        "ingest.browser_export.fetch" in e and "connect_timeout_s" in e and "unknown" in e
        for e in errors
    )


def test_fetch_user_agent_accepted(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["fetch"] = {
        "user_agent": "ExampleOrgDataPlatform/1.0 (contact: data@example.org)"
    }
    assert validate_ingest_contract_schema(data) == []


def test_fetch_user_agent_non_ascii_rejected(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["fetch"] = {"user_agent": "café-bot/1.0"}
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.browser_export.fetch.user_agent" in e for e in errors)


def test_fetch_block_missing_entirely_rejects_missing_user_agent(
    _stub_export_plan: type,
) -> None:
    """Pre-merge review gate finding 7.4: ``fetch`` was previously optional
    in full, so an absent block silently ran the session with chromium's
    default ``HeadlessChrome/...`` User-Agent. Mirrors
    ``api_crawl.rate_limit_rps`` being load-required for the same reason
    -- a contract author must consciously supply the value."""
    data = _browser_export_data()
    del _browser_export_block(data)["fetch"]
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.browser_export.fetch.user_agent" in e and "required" in e for e in errors), (
        errors
    )


def test_fetch_present_without_user_agent_rejected(_stub_export_plan: type) -> None:
    """A ``fetch`` block that sets only unrelated (invalid) keys must still
    report the missing ``user_agent`` -- the requiredness check is
    independent of ``_validate_fetch_block``'s own per-key validation."""
    data = _browser_export_data()
    _browser_export_block(data)["fetch"] = {}
    errors = validate_ingest_contract_schema(data)
    assert any("ingest.browser_export.fetch.user_agent" in e and "required" in e for e in errors), (
        errors
    )


def test_validate_content_content_type_in_rejected_with_reason(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["validate_content"] = {"content_type_in": ["application/zip"]}
    errors = validate_ingest_contract_schema(data)
    matches = [e for e in errors if "content_type_in" in e]
    (msg,) = matches
    assert "is not supported" in msg


def test_validate_content_reject_first_bytes_match_accepted(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["validate_content"] = {
        "reject_first_bytes_match": ["<!DOCTYPE html", "<html"],
        "max_first_bytes_check": 256,
    }
    assert validate_ingest_contract_schema(data) == []


def test_sensitivity_phi_with_browser_export_rejected(_stub_export_plan: type) -> None:
    data = _browser_export_data(sensitivity="phi", data_owner="data-platform")
    errors = validate_ingest_contract_schema(data)
    (msg,) = [e for e in errors if "sensitivity" in e and "browser_export" in e]
    assert "public" in msg
    assert "SECURITY.md" in msg


def test_sensitivity_confidential_with_browser_export_rejected(_stub_export_plan: type) -> None:
    data = _browser_export_data(sensitivity="confidential", data_owner="data-platform")
    errors = validate_ingest_contract_schema(data)
    (msg,) = [e for e in errors if "sensitivity" in e and "browser_export" in e]
    assert "public" in msg
    assert "SECURITY.md" in msg


def test_sensitivity_public_with_browser_export_accepted(_stub_export_plan: type) -> None:
    data = _browser_export_data(sensitivity="public")
    errors = validate_ingest_contract_schema(data)
    assert not any("sensitivity" in e for e in errors)


def test_missing_sensitivity_does_not_double_report_for_browser_export(
    _stub_export_plan: type,
) -> None:
    data = _browser_export_data()
    del data["sensitivity"]
    errors = validate_ingest_contract_schema(data)
    matches = [e for e in errors if "sensitivity" in e]
    assert len(matches) == 1


def test_compliance_review_required_for_browser_export(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    del data["compliance_review"]
    errors = validate_ingest_contract_schema(data)
    assert any("compliance_review" in e and "required" in e for e in errors)


def test_payload_filename_template_rejected_for_multi_file_plan(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["export_plan"] = "_stub_multi_file_export_plan"
    data["ingest"]["payload_filename_template"] = "{source_name}_{partition_key}.csv"  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    (msg,) = [e for e in errors if "payload_filename_template" in e]
    assert "_stub_multi_file_export_plan" in msg
    assert "multi_file" in msg


def test_payload_filename_template_accepted_for_single_file_plan(_stub_export_plan: type) -> None:
    data = _browser_export_data()
    data["ingest"]["payload_filename_template"] = "{source_name}_{partition_key}.csv"  # type: ignore[index]
    assert validate_ingest_contract_schema(data) == []


def test_payload_filename_template_with_unregistered_plan_reports_once(
    _stub_export_plan: type,
) -> None:
    data = _browser_export_data()
    _browser_export_block(data)["export_plan"] = "mystery_plan"
    data["ingest"]["payload_filename_template"] = "{source_name}_{partition_key}.csv"  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    matches = [e for e in errors if "mystery_plan" in e]
    assert len(matches) == 1


def test_http_urls_validate_content_error_strings_unchanged() -> None:
    """Explicit refactor guard for the shared ``validate_content`` validator
    (#463 3B): literals captured against ``origin/main`` before the
    ``(block, known, block_path)`` signature change, so a future edit that
    reorders the prefix construction is caught here rather than only via
    the pre-existing (less exact) substring assertions above."""
    data = _happy_data()
    data["ingest"]["http_urls"]["validate_content"] = {"mystery": "x"}  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert "'ingest.http_urls.validate_content': unknown field 'mystery'." in errors

    data = _happy_data()
    data["ingest"]["http_urls"]["validate_content"] = {"content_type_in": []}  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert "'ingest.http_urls.validate_content'.content_type_in must be a non-empty list" in errors

    data = _happy_data()
    data["ingest"]["http_urls"]["validate_content"] = {"max_first_bytes_check": 0}  # type: ignore[index]
    errors = validate_ingest_contract_schema(data)
    assert (
        "'ingest.http_urls.validate_content'.max_first_bytes_check must be a positive integer"
        in errors
    )
