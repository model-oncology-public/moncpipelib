"""Contract loading and YAML parsing.

This module handles loading data contracts from YAML files, validating
their structure, and converting them to DataContract objects.
"""

from __future__ import annotations

import difflib
import inspect
import logging
import uuid as _uuid_mod
import warnings
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path
from typing import Any

import yaml

from moncpipelib.config import (
    CONTRACT_FILE_PATTERN,
    INGEST_FILE_PATTERN,
    SOURCE_FILE_PATTERN,
    VALID_LAYERS,
)
from moncpipelib.contracts.exceptions import (
    ContractNotFoundError,
    ContractValidationError,
)
from moncpipelib.contracts.models import (
    SLA,
    Column,
    ColumnTest,
    ColumnType,
    ContractCorpus,
    DataContract,
    DataSource,
    FromIngestTemplate,
    IngestContract,
    LineageConfig,
    Owner,
    Period,
    Schema,
    Severity,
    TableExpectation,
    TestingConfig,
    UpstreamDependency,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _IndexedContract:
    """A discovered contract file with the identity fields used for resolution."""

    layer: str | None
    path: Path
    sink_ids: frozenset[str]
    search_root: str


@dataclass(frozen=True)
class _ContractIndex:
    """Layer- and sink-aware contract index (#405).

    ``by_asset`` maps the bare ``asset:`` field to entries; ``by_sink`` maps
    sink-qualified identities (``"schema/table"`` for each table sink that
    declares both) to the same entries, so contracts sharing an asset name
    across schemas remain individually addressable.
    """

    by_asset: dict[str, list[_IndexedContract]]
    by_sink: dict[str, list[_IndexedContract]]


# Contract index cache: maps frozenset of resolved search paths to the built
# _ContractIndex.  Built lazily on first lookup, persists for the lifetime of
# the process (contracts don't change mid-run).
_contract_index_cache: dict[frozenset[str], _ContractIndex] = {}

# Required top-level fields
REQUIRED_FIELDS = {"version", "pipeline_id", "asset", "layer", "schema"}

# Supported contract version
SUPPORTED_VERSIONS = {"1.0"}

# Column test types that don't require parameters
SIMPLE_TESTS = {"not_null", "unique", "not_in_future"}

# Known fields for each structural level — used for unknown-key validation
KNOWN_TOP_LEVEL_FIELDS: frozenset[str] = frozenset(
    {
        "version",
        "pipeline_id",
        "asset",
        "layer",
        "schema",
        "description",
        "owner",
        "expectations",
        "upstream",
        "sla",
        "sources",
        "sinks",
        "testing",
        "lineage",
        "tags",
        "parameters",
        "data_source",
    }
)
KNOWN_PERIOD_FIELDS: frozenset[str] = frozenset(
    {"source", "effective_from", "effective_to", "partition_key"}
)
KNOWN_DATA_SOURCE_FIELDS: frozenset[str] = frozenset(
    {"source_id", "source_name", "description", "periods", "ingest_source"}
)
KNOWN_FROM_INGEST_TEMPLATE_FIELDS: frozenset[str] = frozenset({"source", "effective_from_field"})
KNOWN_FROM_INGEST_PERIODS_FIELDS: frozenset[str] = frozenset({"mode", "template"})

# Known fields for *.ingest.yaml files
KNOWN_INGEST_TOP_LEVEL_FIELDS: frozenset[str] = frozenset(
    {
        "source_id",
        "source_name",
        "sensitivity",
        "ingest",
        "data_owner",
        "compliance_review",
        "description",
    }
)
KNOWN_INGEST_INNER_FIELDS: frozenset[str] = frozenset(
    {
        "pattern",
        "prefix",
        "extract",
        "extract_filter",
        "strip_extensions",
        "payload_filename_template",
        # Pattern-specific blocks are keyed by the pattern name
        # (e.g. "http_urls", "api_resolver") and validated by the
        # pattern implementation rather than by the generic loader.
        "http_urls",
        "api_resolver",
        "api_crawl",
    }
)
KNOWN_SENSITIVITIES: frozenset[str] = frozenset({"public", "confidential", "phi"})
KNOWN_HTTP_URLS_FIELDS: frozenset[str] = frozenset(
    {"idempotency", "fetch", "periods", "validate_content"}
)
KNOWN_VALIDATE_CONTENT_FIELDS: frozenset[str] = frozenset(
    {"content_type_in", "reject_first_bytes_match", "max_first_bytes_check"}
)
KNOWN_API_RESOLVER_FIELDS: frozenset[str] = frozenset(
    {"resolver", "resolver_config", "credential", "partition", "idempotency", "fetch"}
)
KNOWN_API_RESOLVER_CREDENTIAL_FIELDS: frozenset[str] = frozenset({"secret_name"})
KNOWN_API_RESOLVER_PARTITION_FIELDS: frozenset[str] = frozenset({"mode", "key_from"})
KNOWN_API_RESOLVER_PARTITION_MODES: frozenset[str] = frozenset({"dynamic"})
_COMMON_FETCH_FIELDS: frozenset[str] = frozenset(
    {"retries", "timeout_s", "connect_timeout_s", "user_agent"}
)
KNOWN_HTTP_URLS_FETCH_FIELDS: frozenset[str] = _COMMON_FETCH_FIELDS | {"follow_redirects"}
# api_resolver's fetch block intentionally omits follow_redirects: the
# resolved-URL download hardcodes follow_redirects=True (see
# ApiResolverPattern.materialize_partition).
KNOWN_API_RESOLVER_FETCH_FIELDS: frozenset[str] = _COMMON_FETCH_FIELDS
KNOWN_API_CRAWL_FIELDS: frozenset[str] = frozenset(
    {
        "crawl_plan",
        "crawl_config",
        "resolver",
        "resolver_config",
        "credential",
        "partition",
        "fetch",
        "rate_limit_rps",
    }
)
# Same rationale as api_resolver: crawl GETs hardcode
# follow_redirects=True (see ApiCrawlPattern.materialize_partition).
KNOWN_API_CRAWL_FETCH_FIELDS: frozenset[str] = _COMMON_FETCH_FIELDS
KNOWN_HTTP_URLS_PERIOD_FIELDS: frozenset[str] = frozenset(
    {"partition_key", "effective_from", "effective_to", "urls"}
)
KNOWN_SCHEMA_FIELDS: frozenset[str] = frozenset({"columns", "strict"})
KNOWN_COLUMN_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "type",
        "nullable",
        "description",
        "primary_key",
        "managed",
        "pii",
        "phi",
        "tests",
    }
)
KNOWN_OWNER_FIELDS: frozenset[str] = frozenset({"team", "contact", "slack_channel"})
KNOWN_UPSTREAM_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "type",
        "system",
        "description",
    }
)
KNOWN_SLA_FIELDS: frozenset[str] = frozenset(
    {
        "freshness_hours",
        "update_frequency",
        "availability_percent",
    }
)
KNOWN_TESTING_FIELDS: frozenset[str] = frozenset(
    {
        "enabled",
        "source_row_limit",
        "source_where_clause",
        "expected_min_rows",
        "expected_max_rows",
        "timeout_seconds",
    }
)
KNOWN_LINEAGE_FIELDS: frozenset[str] = frozenset(
    {
        "enabled",
        "source_system",
        "transformation_type",
    }
)

# Valid types for source and sink entries
KNOWN_SOURCE_SINK_TYPES: frozenset[str] = frozenset({"table", "external"})

# Known fields per source/sink type.
# External entries are permissive (only 'type' required) and have no field set.
# 'type' is included in both table sets so it is not flagged as unknown.
KNOWN_SOURCE_TABLE_FIELDS: frozenset[str] = frozenset(
    {
        "type",
        "schema",
        "table",
        "database",
        "description",
    }
)
KNOWN_SINK_TABLE_FIELDS: frozenset[str] = frozenset(
    {
        "type",
        "schema",
        "table",
        "database",
        "description",
        "mode",
        "primary_key",
        "partition_column",
        # SCD2 enforcement fields — reconciled at write time by PostgresIOManager
        "business_key",
        "tracked_columns",
        "detect_deletes",
        # Upsert change-guard: suppress DO UPDATE for unchanged rows
        # (mirror issue model-oncology-public/moncpipelib#3)
        "skip_unchanged",
        # Spec'd and consumed by ContractReconciler.reconcile_sequence_column
        # since #109, but missing here until #401 — the strict validator
        # rejected a field the runtime actually honors.
        "sequence_column",
    }
)

# Valid values for sink mode (avoids importing WriteMode from IO manager to prevent
# circular dependency). Must be kept in sync with WriteMode enum.
KNOWN_SINK_MODES: frozenset[str] = frozenset({"full_refresh", "upsert", "append", "scd2"})

# Test modifier keys (valid alongside a test type in dict-form tests)
TEST_MODIFIERS: frozenset[str] = frozenset({"severity", "when"})

# All supported column test types (both simple-string and dict form)
KNOWN_COLUMN_TEST_TYPES: frozenset[str] = frozenset(
    {
        "not_null",
        "unique",
        "not_in_future",
        "accepted_values",
        "not_in",
        "pattern",
        "greater_than",
        "greater_than_or_equal",
        "less_than",
        "less_than_or_equal",
        "between",
        "within_days",
        "min_length",
        "max_length",
    }
)

# All supported table expectation types
KNOWN_EXPECTATION_TYPES: frozenset[str] = frozenset(
    {
        "row_count",
        "freshness",
        "null_percentage",
        "unique_combination",
        "history_completeness",
    }
)

# Legal parameter keys for each table expectation type. Reserved modifier
# keys (``severity``) must be siblings of the type key, never nested inside
# the parameter mapping -- ``_parse_expectation`` copies the nested mapping
# verbatim into ``TableExpectation.parameters`` and reads ``severity`` only
# from the sibling level, so a misplaced key is silently ignored at runtime
# (#394).
KNOWN_EXPECTATION_PARAMS: dict[str, frozenset[str]] = {
    "row_count": frozenset({"min", "max"}),
    "freshness": frozenset({"column", "max_age_hours"}),
    "null_percentage": frozenset({"column", "max_percent"}),
    "unique_combination": frozenset({"columns"}),
    "history_completeness": frozenset(),
}

# Legal parameter keys for each dict-form column test type. Scalar shorthand
# (e.g. ``greater_than: 0``) parses to ``{"value": <scalar>}`` and is always
# valid; this map governs explicit mapping form only. Where the runtime
# dispatchers accept an alias (``values``/``value``, ``days``/``value``,
# ``length``/``value``), both spellings are listed.
KNOWN_COLUMN_TEST_PARAMS: dict[str, frozenset[str]] = {
    "not_null": frozenset(),
    "unique": frozenset(),
    "not_in_future": frozenset(),
    "accepted_values": frozenset({"values", "value"}),
    "not_in": frozenset({"values", "value"}),
    "pattern": frozenset({"value"}),
    "greater_than": frozenset({"value"}),
    "greater_than_or_equal": frozenset({"value"}),
    "less_than": frozenset({"value"}),
    "less_than_or_equal": frozenset({"value"}),
    "between": frozenset({"min", "max"}),
    "within_days": frozenset({"value", "days"}),
    "min_length": frozenset({"value", "length"}),
    "max_length": frozenset({"value", "length"}),
}

# Cross-level field path lookup: maps every known structural field name to every
# dotted path where it can validly appear. Used to hint at correct placement when a
# field is syntactically valid but placed under the wrong section.
_SECTION_PATHS: tuple[tuple[frozenset[str], str], ...] = (
    (KNOWN_TOP_LEVEL_FIELDS, "{field}"),
    (KNOWN_SCHEMA_FIELDS, "schema.{field}"),
    (KNOWN_COLUMN_FIELDS, "schema.columns[*].{field}"),
    (KNOWN_OWNER_FIELDS, "owner.{field}"),
    (KNOWN_UPSTREAM_FIELDS, "upstream[*].{field}"),
    (KNOWN_SLA_FIELDS, "sla.{field}"),
    (KNOWN_TESTING_FIELDS, "testing.{field}"),
    (KNOWN_LINEAGE_FIELDS, "lineage.{field}"),
)
_FIELD_TO_PATHS: dict[str, list[str]] = {}
for _known_set, _path_template in _SECTION_PATHS:
    for _f in sorted(_known_set):
        _FIELD_TO_PATHS.setdefault(_f, []).append(_path_template.format(field=_f))


def _suggest(unknown: str, known: frozenset[str]) -> str:
    """Return a same-level 'did you mean' hint if a close match exists, else empty."""
    matches = difflib.get_close_matches(unknown, known, n=1, cutoff=0.6)
    return f" Did you mean '{matches[0]}'?" if matches else ""


def _suggest_path(unknown: str) -> str:
    """Return a cross-level path hint if the field is known to exist elsewhere."""
    paths = _FIELD_TO_PATHS.get(unknown)
    if not paths:
        return ""
    if len(paths) == 1:
        return f" Did you mean '{paths[0]}'?"
    return f" This field is valid under: {', '.join(paths)}."


def _check_unknown_keys(d: dict[Any, Any], known: frozenset[str], prefix: str) -> list[str]:
    """Return errors for any keys in d that are not in the known set."""
    unknown_keys = sorted(str(k) for k in d if str(k) not in known)
    return [
        f"{prefix}: unknown field '{k}'.{_suggest(k, known) or _suggest_path(k)}"
        for k in unknown_keys
    ]


def _validate_source_sink_entry(
    entry: Any,
    index: int,
    label: str,
    table_fields: frozenset[str],
) -> list[str]:
    """Validate a single source or sink entry by its declared type.

    Args:
        entry: The raw source/sink dict from YAML.
        index: Position in the sources/sinks list (for error messages).
        label: "Source" or "Sink" (for error messages).
        table_fields: Allowed fields for type='table' entries — differs between
            sources (no 'mode') and sinks ('mode' is valid).
    """
    errors: list[str] = []
    prefix = f"{label} {index}"

    if not isinstance(entry, dict):
        return [f"{prefix}: must be an object"]

    entry_type = entry.get("type")
    if not entry_type:
        errors.append(f"{prefix}: 'type' is required")
        return errors

    if str(entry_type) not in KNOWN_SOURCE_SINK_TYPES:
        errors.append(
            f"{prefix}: unknown type '{entry_type}'."
            f"{_suggest(str(entry_type), KNOWN_SOURCE_SINK_TYPES)}"
        )
        return errors  # field validation is type-dependent; skip without a known type

    if str(entry_type) == "table":
        errors.extend(_check_unknown_keys(entry, table_fields, prefix))
        for req_field in ("schema", "table"):
            if req_field not in entry:
                errors.append(f"{prefix}: '{req_field}' is required for type 'table'")

        # Validate sink-specific field types
        if label == "Sink":
            mode = entry.get("mode")
            if mode is not None and str(mode) not in KNOWN_SINK_MODES:
                errors.append(
                    f"{prefix}: unknown mode '{mode}'.{_suggest(str(mode), KNOWN_SINK_MODES)}"
                )

            for list_field in ("business_key", "tracked_columns", "primary_key"):
                val = entry.get(list_field)
                if val is not None:
                    if isinstance(val, str):
                        pass  # bare string is valid (normalised to list at write time)
                    elif isinstance(val, list):
                        for j, item in enumerate(val):
                            if not isinstance(item, str):
                                errors.append(
                                    f"{prefix}: '{list_field}[{j}]' must be a string, "
                                    f"got {type(item).__name__}"
                                )
                    else:
                        errors.append(
                            f"{prefix}: '{list_field}' must be a string or list of strings"
                        )

            sc = entry.get("sequence_column")
            if "sequence_column" in entry and sc is not None and not isinstance(sc, str):
                errors.append(
                    f"{prefix}: 'sequence_column' must be a string or null, got {type(sc).__name__}"
                )

            dd = entry.get("detect_deletes")
            if dd is not None and not isinstance(dd, bool):
                errors.append(
                    f"{prefix}: 'detect_deletes' must be a boolean, got {type(dd).__name__}"
                )
            elif dd and mode is not None and str(mode) in KNOWN_SINK_MODES and str(mode) != "scd2":
                # detect_deletes is consumed only by the SCD2 write path; on any
                # other declared mode it is inert config that reads as if delete
                # detection were active (#401).
                errors.append(
                    f"{prefix}: 'detect_deletes' is only valid with mode 'scd2', got mode '{mode}'"
                )

            su = entry.get("skip_unchanged")
            if su is not None and not isinstance(su, bool):
                errors.append(
                    f"{prefix}: 'skip_unchanged' must be a boolean, got {type(su).__name__}"
                )
            elif (
                su and mode is not None and str(mode) in KNOWN_SINK_MODES and str(mode) != "upsert"
            ):
                # skip_unchanged is consumed only by the upsert merge; on any
                # other declared mode it is inert config that reads as if
                # unchanged-row suppression were active.
                errors.append(
                    f"{prefix}: 'skip_unchanged' is only valid with mode 'upsert', got mode "
                    f"'{mode}'"
                )
    # external entries are permissive: only 'type' is required; additional fields
    # vary by system and are not validated.

    return errors


def validate_data_source_schema(data: dict[str, Any]) -> list[str]:
    """Validate data source YAML structure.

    Args:
        data: Parsed YAML dictionary from a ``*.source.yaml`` file.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []

    # Check for unknown top-level fields
    errors.extend(_check_unknown_keys(data, KNOWN_DATA_SOURCE_FIELDS, "DataSource"))

    # Required fields
    if "source_id" not in data:
        errors.append("Missing required field: 'source_id'")
    elif not isinstance(data["source_id"], str):
        errors.append("'source_id' must be a string")
    else:
        try:
            _uuid_mod.UUID(str(data["source_id"]))
        except ValueError:
            errors.append(f"'source_id' must be a valid UUID, got '{data['source_id']}'")

    if "source_name" not in data:
        errors.append("Missing required field: 'source_name'")
    elif not isinstance(data["source_name"], str):
        errors.append("'source_name' must be a string")

    if "ingest_source" in data and not isinstance(data["ingest_source"], str):
        errors.append("'ingest_source' must be a string")

    if "periods" not in data:
        errors.append("Missing required field: 'periods'")
    elif isinstance(data["periods"], dict):
        errors.extend(_validate_from_ingest_periods(data["periods"], data))
    elif isinstance(data["periods"], list):
        prev_to: _date | None = None
        open_ended_count = 0
        for i, period in enumerate(data["periods"]):
            prefix = f"Period {i}"
            if not isinstance(period, dict):
                errors.append(f"{prefix}: must be an object")
                continue
            errors.extend(_check_unknown_keys(period, KNOWN_PERIOD_FIELDS, f"'{prefix}'"))
            if "source" not in period:
                errors.append(f"{prefix}: missing required field 'source'")
            if "effective_from" not in period:
                errors.append(f"{prefix}: missing required field 'effective_from'")
                continue
            eff_from = period.get("effective_from")
            eff_to = period.get("effective_to")
            if not isinstance(eff_from, _date):
                errors.append(
                    f"{prefix}: 'effective_from' must be a date (YAML date format: YYYY-MM-DD)"
                )
                continue
            if eff_to is not None and not isinstance(eff_to, _date):
                errors.append(f"{prefix}: 'effective_to' must be a date or null")
                continue
            if eff_to is None:
                open_ended_count += 1
            if prev_to is not None and isinstance(eff_from, _date) and eff_from < prev_to:
                errors.append(
                    f"{prefix}: effective_from ({eff_from}) overlaps "
                    f"with previous period's effective_to ({prev_to})"
                )
            prev_to = eff_to if isinstance(eff_to, _date) else None
        if open_ended_count > 1:
            errors.append(
                "At most one period may have effective_to: null (the current/open-ended period)"
            )
    else:
        errors.append("'periods' must be a list or a {mode: from_ingest, template: {...}} mapping")

    return errors


def _validate_from_ingest_periods(periods: dict[str, Any], data: dict[str, Any]) -> list[str]:
    """Validate a ``periods: {mode: from_ingest, template: {...}}`` block.

    Returns a list of error messages; empty on success.
    """
    errors: list[str] = []
    errors.extend(_check_unknown_keys(periods, KNOWN_FROM_INGEST_PERIODS_FIELDS, "'periods'"))
    mode = periods.get("mode")
    if mode != "from_ingest":
        errors.append(
            f"'periods.mode' must be 'from_ingest' when periods is a mapping, got '{mode}'"
        )
    if data.get("ingest_source") is None:
        errors.append(
            "'periods.mode: from_ingest' requires 'ingest_source' to be set on the data source"
        )
    template = periods.get("template")
    if template is None:
        errors.append("'periods.template' is required for mode 'from_ingest'")
        return errors
    if not isinstance(template, dict):
        errors.append("'periods.template' must be a mapping")
        return errors
    errors.extend(
        _check_unknown_keys(template, KNOWN_FROM_INGEST_TEMPLATE_FIELDS, "'periods.template'")
    )
    for req in ("source", "effective_from_field"):
        if req not in template:
            errors.append(f"'periods.template': missing required field '{req}'")
        elif not isinstance(template[req], str):
            errors.append(f"'periods.template.{req}' must be a string")
    return errors


def validate_ingest_contract_schema(data: dict[str, Any]) -> list[str]:
    """Validate a ``*.ingest.yaml`` file's top-level structure.

    Args:
        data: Parsed YAML dictionary.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []

    errors.extend(_check_unknown_keys(data, KNOWN_INGEST_TOP_LEVEL_FIELDS, "IngestContract"))

    if "source_id" not in data:
        errors.append("Missing required field: 'source_id'")
    elif not isinstance(data["source_id"], str):
        errors.append("'source_id' must be a string")
    else:
        try:
            _uuid_mod.UUID(str(data["source_id"]))
        except ValueError:
            errors.append(f"'source_id' must be a valid UUID, got '{data['source_id']}'")

    if "source_name" not in data:
        errors.append("Missing required field: 'source_name'")
    elif not isinstance(data["source_name"], str):
        errors.append("'source_name' must be a string")

    sensitivity = data.get("sensitivity")
    if sensitivity is None:
        errors.append("Missing required field: 'sensitivity'")
    elif sensitivity not in KNOWN_SENSITIVITIES:
        errors.append(
            f"'sensitivity' must be one of {sorted(KNOWN_SENSITIVITIES)}, got '{sensitivity}'"
        )

    # Attestation rule: phi / confidential contracts require data_owner + compliance_review.
    if sensitivity in {"phi", "confidential"}:
        for req in ("data_owner", "compliance_review"):
            if not data.get(req):
                errors.append(f"'{req}' is required when sensitivity is '{sensitivity}'")

    ingest = data.get("ingest")
    if ingest is None:
        errors.append("Missing required field: 'ingest'")
        return errors
    if not isinstance(ingest, dict):
        errors.append("'ingest' must be a mapping")
        return errors

    errors.extend(_check_unknown_keys(ingest, KNOWN_INGEST_INNER_FIELDS, "'ingest'"))

    pattern = ingest.get("pattern")
    if pattern is None:
        errors.append("'ingest.pattern' is required")
    elif not isinstance(pattern, str):
        errors.append("'ingest.pattern' must be a string")

    prefix = ingest.get("prefix")
    if prefix is None:
        errors.append("'ingest.prefix' is required")
    elif not isinstance(prefix, str):
        errors.append("'ingest.prefix' must be a string")

    extract = ingest.get("extract")
    if extract is not None and (
        not isinstance(extract, list) or not all(isinstance(e, str) for e in extract)
    ):
        errors.append("'ingest.extract' must be a list of strings")

    strip = ingest.get("strip_extensions")
    if strip is not None and (
        not isinstance(strip, list) or not all(isinstance(e, str) for e in strip)
    ):
        errors.append("'ingest.strip_extensions' must be a list of strings")

    errors.extend(_validate_ingest_extract_filter(ingest))
    errors.extend(_validate_payload_filename_template(ingest))

    # Pattern-specific validation. Each pattern owns its own block validator.
    if pattern == "http_urls":
        errors.extend(_validate_http_urls_block(ingest.get("http_urls")))
        errors.extend(_validate_non_archive_filename_uniqueness(ingest))
    elif pattern == "api_resolver":
        errors.extend(_validate_api_resolver_block(ingest.get("api_resolver")))
    elif pattern == "api_crawl":
        errors.extend(_validate_api_crawl_block(ingest.get("api_crawl")))

    return errors


def _validate_ingest_extract_filter(ingest: dict[str, Any]) -> list[str]:
    """Validate the optional ``ingest.extract_filter`` field (ADR-1).

    Rules (per
    ``docs/migrations/20260426_phase2-ingest-decisions.md``):

    1. Field is optional; when absent, the contract preserves Phase 1
       "extract everything" behavior.
    2. When present, must be a non-empty list of non-empty strings.
       ``extract_filter: []`` is rejected -- treating empty as
       "no files" would silently extract nothing, a footgun. Authors
       who mean "extract everything" omit the field entirely.
    3. ``extract_filter`` requires ``extract`` to also be present;
       filtering only matters when extraction is happening.
    """
    errors: list[str] = []
    if "extract_filter" not in ingest:
        return errors

    value = ingest.get("extract_filter")
    if not isinstance(value, list):
        errors.append("'ingest.extract_filter' must be a list of strings")
        return errors
    if not value:
        errors.append(
            "'ingest.extract_filter' must be a non-empty list "
            "(omit the field to extract everything)"
        )
        return errors
    if not all(isinstance(p, str) and p for p in value):
        errors.append("'ingest.extract_filter' must be a list of non-empty fnmatch patterns")
        return errors
    if "extract" not in ingest:
        errors.append(
            "'ingest.extract_filter' requires 'ingest.extract' to be set "
            "(the filter only applies during extraction)"
        )

    return errors


def _validate_payload_filename_template(ingest: dict[str, Any]) -> list[str]:
    """Validate the optional ``ingest.payload_filename_template`` field (#270).

    The template is rendered through the same bounded placeholder set
    as ``prefix_template``; the per-placeholder check happens at
    materialize time via :func:`~moncpipelib.ingest.prefix.render_payload_filename`.
    Loader-time validation only confirms the field is a non-empty string
    when present.
    """
    if "payload_filename_template" not in ingest:
        return []
    value = ingest.get("payload_filename_template")
    if not isinstance(value, str) or not value:
        return ["'ingest.payload_filename_template' must be a non-empty string"]
    return []


def _validate_non_archive_filename_uniqueness(ingest: dict[str, Any]) -> list[str]:
    """Multi-URL non-archive periods must produce unique landed filenames (#270).

    When ``extract: []`` (no archive expansion), a period with multiple
    URLs would silently land them all under the same precedence-derived
    filename (since templates render identically for a given partition,
    and the URL-basename fallback is the only per-URL signal we can
    compute without a live HTTP fetch).  Catching this at intake
    prevents a real non-determinism bug from shipping.

    Skipped when:

    - The contract is not ``http_urls`` (only checked at the
      ``http_urls`` call site upstream).
    - ``extract`` is non-empty (archive members keep their in-archive
      names, no collision possible).
    - Any single period has only one URL.

    The check operates on the contract + URLs only -- the
    ``Content-Disposition`` precedence level is excluded since
    validation does not fetch.  Authors fix collisions by:

    - Setting ``payload_filename_template`` with a partition-or-period
      token that guarantees uniqueness (in practice, since templates
      render to the same string within a period, this only helps for
      single-URL periods today; the broader follow-on is per-URL
      filename overrides), OR
    - Writing distinct URLs whose sanitized basenames differ.

    Imports of ingest helpers are lazy so contract-only consumers
    don't pull in :mod:`moncpipelib.ingest` at load time.
    """
    extract = ingest.get("extract")
    if extract:
        return []

    block = ingest.get("http_urls")
    if not isinstance(block, dict):
        return []
    periods = block.get("periods")
    if not isinstance(periods, list):
        return []

    # Lazy imports: keep the loader importable from contract-only
    # consumers without forcing the ingest package to load.  Mirrors
    # the existing pattern for the resolver registry.
    from pathlib import PurePosixPath
    from urllib.parse import urlparse

    from moncpipelib.ingest.filenames import sanitize_blob_filename
    from moncpipelib.ingest.prefix import _ALLOWED_PLACEHOLDERS

    template = ingest.get("payload_filename_template")
    if isinstance(template, str):
        # Confirm the template only references known placeholders so
        # we can reason about whether it disambiguates URLs (it cannot
        # today: both placeholders are fixed within a period).  The
        # template engine itself enforces the same rule at render
        # time; validating here as a defensive double-check.
        unknown = _scan_template_placeholders(template) - _ALLOWED_PLACEHOLDERS
        if unknown:
            # Render-time error will catch this; skip the uniqueness
            # check to avoid a confusing duplicate diagnostic.
            return []

    errors: list[str] = []
    for i, period in enumerate(periods):
        if not isinstance(period, dict):
            continue
        urls = period.get("urls")
        if not isinstance(urls, list) or len(urls) <= 1:
            continue
        # Compute the per-URL landed filename: template overrides
        # everything when set, else fall back to the sanitized URL
        # basename.  Within a single period, partition_key + source_name
        # are fixed so the template renders identically -- if a template
        # is set, EVERY url in this period collides on the same name.
        names: list[str | None] = []
        if isinstance(template, str):
            # All URLs collide on the rendered template; surface as a
            # single collision row.  We keep the same data structure
            # as the URL-basename branch for uniformity.
            names = [template] * len(urls)
        else:
            for url in urls:
                if not isinstance(url, str):
                    names.append(None)
                    continue
                basename = PurePosixPath(urlparse(url).path).name
                names.append(sanitize_blob_filename(basename))

        seen: dict[str | None, list[int]] = {}
        for idx, name in enumerate(names):
            seen.setdefault(name, []).append(idx)
        collisions = {n: idxs for n, idxs in seen.items() if len(idxs) > 1}
        if collisions:
            for name, idxs in sorted(collisions.items(), key=lambda kv: (kv[0] or "", kv[1])):
                errors.append(
                    f"'ingest.http_urls.periods[{i}]': non-archive period has "
                    f"multiple URLs that resolve to the same landed filename "
                    f"{name!r} (url indices {idxs}). Set distinct URLs whose "
                    f"basenames differ, or stop bundling these URLs in one "
                    f"period."
                )
    return errors


def _scan_template_placeholders(template: str) -> set[str]:
    """Return the set of ``{...}`` placeholder names referenced by ``template``."""
    import re

    return set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", template))


def _validate_fetch_block(fetch: Any, known: frozenset[str], block_path: str) -> list[str]:
    """Validate an optional ``fetch`` sub-block shared by both ingest patterns.

    Checks the key set against the pattern's ``known`` fields and the
    value types of the common knobs, so a bad value fails at contract
    load instead of deep inside a materialization.  Pattern-specific
    keys (http_urls' ``follow_redirects``) are type-checked by the
    pattern's own validator after this shared pass.
    """
    errors: list[str] = []
    if fetch is None:
        return errors
    prefix = f"'{block_path}.fetch'"
    if not isinstance(fetch, dict):
        errors.append(f"{prefix} must be a mapping")
        return errors

    errors.extend(_check_unknown_keys(fetch, known, prefix))
    if "retries" in fetch:
        value = fetch["retries"]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"'{block_path}.fetch.retries' must be a non-negative integer")
    for key in ("timeout_s", "connect_timeout_s"):
        if key in fetch:
            value = fetch[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                errors.append(f"'{block_path}.fetch.{key}' must be a positive number")
    if "user_agent" in fetch:
        value = fetch["user_agent"]
        if not isinstance(value, str) or not value.strip():
            errors.append(f"'{block_path}.fetch.user_agent' must be a non-empty string")
        elif not (value.isascii() and value.isprintable()):
            # httpx encodes header values as strict ASCII and a real
            # transport rejects control characters (e.g. CRLF) only at
            # request time -- catch both at contract load instead.
            errors.append(f"'{block_path}.fetch.user_agent' must be printable ASCII")
    return errors


def _validate_http_urls_block(block: Any) -> list[str]:
    """Validate the ``ingest.http_urls`` inner block."""
    errors: list[str] = []
    if block is None:
        errors.append("'ingest.http_urls' is required when pattern is 'http_urls'")
        return errors
    if not isinstance(block, dict):
        errors.append("'ingest.http_urls' must be a mapping")
        return errors

    errors.extend(_check_unknown_keys(block, KNOWN_HTTP_URLS_FIELDS, "'ingest.http_urls'"))

    fetch = block.get("fetch")
    errors.extend(_validate_fetch_block(fetch, KNOWN_HTTP_URLS_FETCH_FIELDS, "ingest.http_urls"))
    if (
        isinstance(fetch, dict)
        and "follow_redirects" in fetch
        and not isinstance(fetch["follow_redirects"], bool)
    ):
        errors.append("'ingest.http_urls.fetch.follow_redirects' must be a boolean")

    errors.extend(_validate_validate_content_block(block.get("validate_content")))

    periods = block.get("periods")
    if periods is None:
        errors.append("'ingest.http_urls.periods' is required")
        return errors
    if not isinstance(periods, list) or not periods:
        errors.append("'ingest.http_urls.periods' must be a non-empty list")
        return errors

    for i, period in enumerate(periods):
        prefix = f"'ingest.http_urls.periods[{i}]'"
        if not isinstance(period, dict):
            errors.append(f"{prefix}: must be a mapping")
            continue
        errors.extend(_check_unknown_keys(period, KNOWN_HTTP_URLS_PERIOD_FIELDS, prefix))
        if "partition_key" not in period:
            errors.append(f"{prefix}: 'partition_key' is required")
        urls = period.get("urls")
        if urls is None:
            errors.append(f"{prefix}: 'urls' is required")
        elif not isinstance(urls, list) or not urls:
            errors.append(f"{prefix}: 'urls' must be a non-empty list")
        elif not all(isinstance(u, str) for u in urls):
            errors.append(f"{prefix}: 'urls' must be a list of strings")

    return errors


def _validate_validate_content_block(block: Any) -> list[str]:
    """Validate the optional ``ingest.http_urls.validate_content`` block (#228).

    The block is optional; when set, it switches the URL list semantics
    for the period from "process all as union" to "try in order, take
    first valid". See the migration tracker at
    ``docs/migrations/20260426_228-validate-content-and-historical-release.md``.
    """
    errors: list[str] = []
    if block is None:
        return errors
    prefix = "'ingest.http_urls.validate_content'"
    if not isinstance(block, dict):
        errors.append(f"{prefix} must be a mapping")
        return errors

    errors.extend(_check_unknown_keys(block, KNOWN_VALIDATE_CONTENT_FIELDS, prefix))

    for list_key in ("content_type_in", "reject_first_bytes_match"):
        value = block.get(list_key)
        if value is None:
            continue
        if not isinstance(value, list) or not value:
            errors.append(f"{prefix}.{list_key} must be a non-empty list")
            continue
        if not all(isinstance(v, str) and v for v in value):
            errors.append(f"{prefix}.{list_key} must be a list of non-empty strings")

    if "max_first_bytes_check" in block:
        value = block["max_first_bytes_check"]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"{prefix}.max_first_bytes_check must be a positive integer")

    return errors


def _validate_api_resolver_block(block: Any) -> list[str]:
    """Validate the ``ingest.api_resolver`` inner block (ADR-2).

    The generic loader validates the structural shape (resolver name,
    credential block, partition block) and dispatches the
    ``resolver_config`` sub-block to the named resolver's
    :meth:`~moncpipelib.ingest.resolvers.ReleaseResolver.validate_config`
    so per-resolver schema checks fail at contract-load time.
    """
    errors: list[str] = []
    if block is None:
        errors.append("'ingest.api_resolver' is required when pattern is 'api_resolver'")
        return errors
    if not isinstance(block, dict):
        errors.append("'ingest.api_resolver' must be a mapping")
        return errors

    errors.extend(_check_unknown_keys(block, KNOWN_API_RESOLVER_FIELDS, "'ingest.api_resolver'"))

    # Per #413 the fetch sub-block gets the same key/type validation as
    # http_urls (previously its contents were unchecked, so a typo'd knob
    # silently fell through to defaults).
    errors.extend(
        _validate_fetch_block(
            block.get("fetch"), KNOWN_API_RESOLVER_FETCH_FIELDS, "ingest.api_resolver"
        )
    )

    errors.extend(_validate_resolver_ref(block, "ingest.api_resolver"))
    errors.extend(_validate_credential_block(block, "ingest.api_resolver"))
    errors.extend(_validate_partition_block(block, "ingest.api_resolver"))

    return errors


def _validate_resolver_ref(block: dict[str, Any], block_path: str) -> list[str]:
    """Validate ``resolver`` + dispatch ``resolver_config`` to the named
    resolver's ``validate_config``.

    Shared by both resolver-backed patterns (``api_resolver``,
    ``api_crawl`` per #415); ``block_path`` parameterizes error messages
    only.
    """
    errors: list[str] = []
    resolver_name = block.get("resolver")
    resolver = None
    if resolver_name is None:
        errors.append(f"'{block_path}.resolver' is required")
    elif not isinstance(resolver_name, str):
        errors.append(f"'{block_path}.resolver' must be a string")
    else:
        # Lazy import: the contracts package must not require the ingest
        # package at module import time (cookbook tests / lightweight
        # contract-only consumers should still work).
        from moncpipelib.ingest.resolvers import get_resolver

        try:
            resolver = get_resolver(resolver_name)
        except KeyError as e:
            errors.append(f"'{block_path}.resolver': {e}")

    cfg = block.get("resolver_config")
    if cfg is None:
        cfg = {}
    elif not isinstance(cfg, dict):
        errors.append(f"'{block_path}.resolver_config' must be a mapping")
        cfg = {}

    if resolver is not None and isinstance(cfg, dict):
        errors.extend(f"'{block_path}.resolver_config.{e}'" for e in resolver.validate_config(cfg))

    return errors


def _validate_credential_block(block: dict[str, Any], block_path: str) -> list[str]:
    """Validate the optional ``credential`` sub-block (shared shape).

    Optional: contracts against public / unauthenticated APIs (e.g. the
    calendar resolver, RxClass) omit it.  When present, the inner shape
    is still validated.
    """
    errors: list[str] = []
    credential = block.get("credential")
    if credential is None:
        return errors
    if not isinstance(credential, dict):
        errors.append(f"'{block_path}.credential' must be a mapping")
        return errors
    errors.extend(
        _check_unknown_keys(
            credential,
            KNOWN_API_RESOLVER_CREDENTIAL_FIELDS,
            f"'{block_path}.credential'",
        )
    )
    secret_name = credential.get("secret_name")
    if secret_name is None:
        errors.append(f"'{block_path}.credential.secret_name' is required")
    elif not isinstance(secret_name, str) or not secret_name:
        errors.append(f"'{block_path}.credential.secret_name' must be a non-empty string")
    return errors


def _validate_partition_block(block: dict[str, Any], block_path: str) -> list[str]:
    """Validate the required ``partition`` sub-block (shared shape)."""
    errors: list[str] = []
    partition = block.get("partition")
    if partition is None:
        errors.append(f"'{block_path}.partition' is required")
    elif not isinstance(partition, dict):
        errors.append(f"'{block_path}.partition' must be a mapping")
    else:
        errors.extend(
            _check_unknown_keys(
                partition,
                KNOWN_API_RESOLVER_PARTITION_FIELDS,
                f"'{block_path}.partition'",
            )
        )
        mode = partition.get("mode")
        if mode is None:
            errors.append(f"'{block_path}.partition.mode' is required")
        elif mode not in KNOWN_API_RESOLVER_PARTITION_MODES:
            known = sorted(KNOWN_API_RESOLVER_PARTITION_MODES)
            errors.append(f"'{block_path}.partition.mode' must be one of {known}")
        if "key_from" not in partition:
            errors.append(f"'{block_path}.partition.key_from' is required")
        elif not isinstance(partition["key_from"], str) or not partition["key_from"]:
            errors.append(f"'{block_path}.partition.key_from' must be a non-empty string")

    return errors


def _validate_api_crawl_block(block: Any) -> list[str]:
    """Validate the ``ingest.api_crawl`` inner block (#415).

    Structural shape mirrors ``api_resolver`` (resolver-backed
    discovery: shared resolver / credential / partition sub-validators)
    plus the crawl-specific fields: a registered
    :class:`~moncpipelib.ingest.crawl_plans.CrawlPlan` whose
    ``validate_config`` gets the ``crawl_config`` sub-block, and the
    required ``rate_limit_rps`` budget.
    """
    errors: list[str] = []
    if block is None:
        errors.append("'ingest.api_crawl' is required when pattern is 'api_crawl'")
        return errors
    if not isinstance(block, dict):
        errors.append("'ingest.api_crawl' must be a mapping")
        return errors

    errors.extend(_check_unknown_keys(block, KNOWN_API_CRAWL_FIELDS, "'ingest.api_crawl'"))
    errors.extend(
        _validate_fetch_block(block.get("fetch"), KNOWN_API_CRAWL_FETCH_FIELDS, "ingest.api_crawl")
    )

    # crawl plan name + per-plan validate_config dispatch (mirrors the
    # resolver dispatch in _validate_resolver_ref).
    plan_name = block.get("crawl_plan")
    plan = None
    if plan_name is None:
        errors.append("'ingest.api_crawl.crawl_plan' is required")
    elif not isinstance(plan_name, str):
        errors.append("'ingest.api_crawl.crawl_plan' must be a string")
    else:
        # Lazy import for the same contract-only-consumer reason as
        # the resolver registry above.
        from moncpipelib.ingest.crawl_plans import get_crawl_plan

        try:
            plan = get_crawl_plan(plan_name)
        except KeyError as e:
            errors.append(f"'ingest.api_crawl.crawl_plan': {e}")

    crawl_cfg = block.get("crawl_config")
    if crawl_cfg is None:
        crawl_cfg = {}
    elif not isinstance(crawl_cfg, dict):
        errors.append("'ingest.api_crawl.crawl_config' must be a mapping")
        crawl_cfg = {}
    if plan is not None and isinstance(crawl_cfg, dict):
        errors.extend(
            f"'ingest.api_crawl.crawl_config.{e}'" for e in plan.validate_config(crawl_cfg)
        )

    errors.extend(_validate_resolver_ref(block, "ingest.api_crawl"))
    errors.extend(_validate_credential_block(block, "ingest.api_crawl"))
    errors.extend(_validate_partition_block(block, "ingest.api_crawl"))

    # rate_limit_rps: required with guidance -- per maintainer review on
    # the #415 plan, a missing budget must explain WHY the field exists
    # and how to choose a value, not emit a bare "is required".
    if "rate_limit_rps" not in block:
        errors.append(
            "'ingest.api_crawl.rate_limit_rps' is required: api_crawl issues "
            "many requests against a live API, so contracts must declare an "
            "explicit requests-per-second budget at or below the upstream's "
            "published cap (e.g. RxNav allows 20 req/s per IP; a conservative "
            "value such as 5 is typical). Set a number > 0."
        )
    else:
        value = block["rate_limit_rps"]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            errors.append(
                "'ingest.api_crawl.rate_limit_rps' must be a number > 0 "
                "(requests per second, held at or below the upstream's "
                f"published cap); got {value!r}"
            )

    return errors


def load_data_source(path: str | Path) -> DataSource:
    """Load a data source definition from a ``*.source.yaml`` file.

    Args:
        path: Path to the data source YAML file.

    Returns:
        Parsed and validated DataSource.

    Raises:
        ContractNotFoundError: If file does not exist.
        ContractValidationError: If YAML structure is invalid.
    """
    path = Path(path)

    if not path.exists():
        raise ContractNotFoundError(f"Data source file not found: {path}")

    try:
        with path.open() as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ContractValidationError(f"Invalid YAML in {path}: {e}") from e

    if data is None:
        raise ContractValidationError(f"Empty data source file: {path}")

    errors = validate_data_source_schema(data)
    if errors:
        raise ContractValidationError(
            f"Data source validation failed for {path}:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return DataSource(
        source_id=str(data["source_id"]),
        source_name=str(data["source_name"]),
        periods=_parse_periods(data.get("periods")),
        ingest_source=data.get("ingest_source"),
        description=data.get("description"),
    )


def load_ingest_contract(path: str | Path) -> IngestContract:
    """Load an ingest contract from a ``*.ingest.yaml`` file.

    Ingest contracts describe how external data is pulled into the
    blob-landing boundary. One ingest contract feeds one or more
    downstream ``DataSource`` contracts via ``DataSource.ingest_source``.

    Args:
        path: Path to the ingest YAML file.

    Returns:
        Parsed and validated ``IngestContract``.

    Raises:
        ContractNotFoundError: If file does not exist.
        ContractValidationError: If YAML structure is invalid or fails
            attestation / field-type rules.
    """
    path = Path(path)

    if not path.exists():
        raise ContractNotFoundError(f"Ingest contract file not found: {path}")

    try:
        with path.open() as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ContractValidationError(f"Invalid YAML in {path}: {e}") from e

    if data is None:
        raise ContractValidationError(f"Empty ingest contract file: {path}")

    errors = validate_ingest_contract_schema(data)
    if errors:
        raise ContractValidationError(
            f"Ingest contract validation failed for {path}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    ingest = data["ingest"]
    pattern = str(ingest["pattern"])
    # The pattern_config is the inner block keyed by the pattern name.
    # Pattern implementations receive this dict verbatim.
    pattern_config = dict(ingest.get(pattern, {}))

    return IngestContract(
        source_id=str(data["source_id"]),
        source_name=str(data["source_name"]),
        sensitivity=data["sensitivity"],
        pattern=pattern,
        prefix_template=str(ingest["prefix"]),
        extract=tuple(ingest.get("extract") or ()),
        strip_extensions=tuple(ingest.get("strip_extensions") or ()),
        extract_filter=tuple(ingest.get("extract_filter") or ()),
        pattern_config=pattern_config,
        data_owner=data.get("data_owner"),
        compliance_review=data.get("compliance_review"),
        description=data.get("description"),
        payload_filename_template=ingest.get("payload_filename_template"),
    )


def load_all_contracts(root_path: str | Path) -> ContractCorpus:
    """Load and cross-validate every contract under ``root_path``.

    Walks the directory recursively collecting every ``*.ingest.yaml``
    and ``*.source.yaml`` file, loads each via its per-file loader, and
    runs cross-contract validators:

    1. Every ``DataSource.ingest_source`` resolves to a known ingest
       contract.
    2. Static alignment: when ``DataSource.periods`` is a list, every
       ``period.partition_key`` must appear in the linked ingest's
       ``pattern_config["periods"]``.
    3. Dynamic linkage: when ``DataSource.periods`` is a
       ``FromIngestTemplate``, the linked ingest must declare
       ``partition.mode: dynamic``.

    Args:
        root_path: Directory to scan recursively.

    Returns:
        A ``ContractCorpus`` with all loaded contracts.

    Raises:
        ContractNotFoundError: If ``root_path`` does not exist.
        ContractValidationError: If any per-file validation or any
            cross-contract rule fails. All errors are aggregated into a
            single exception message.
    """
    root = Path(root_path)
    if not root.exists():
        raise ContractNotFoundError(f"Contract root not found: {root}")
    if not root.is_dir():
        raise ContractValidationError(f"Contract root is not a directory: {root}")

    ingests: dict[str, IngestContract] = {}
    sources: dict[str, DataSource] = {}
    errors: list[str] = []

    for ingest_file in sorted(root.rglob(INGEST_FILE_PATTERN)):
        try:
            contract = load_ingest_contract(ingest_file)
        except ContractValidationError as e:
            errors.append(str(e))
            continue
        if contract.source_name in ingests:
            errors.append(
                f"Duplicate ingest contract source_name '{contract.source_name}': "
                f"{ingest_file} collides with an earlier load"
            )
            continue
        ingests[contract.source_name] = contract

    for source_file in sorted(root.rglob(SOURCE_FILE_PATTERN)):
        try:
            source = load_data_source(source_file)
        except ContractValidationError as e:
            errors.append(str(e))
            continue
        if source.source_name in sources:
            errors.append(
                f"Duplicate data source source_name '{source.source_name}': "
                f"{source_file} collides with an earlier load"
            )
            continue
        sources[source.source_name] = source

    errors.extend(_validate_cross_contracts(ingests, sources))

    if errors:
        raise ContractValidationError(
            f"Contract corpus validation failed under {root}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return ContractCorpus(ingests=ingests, sources=sources)


def _validate_cross_contracts(
    ingests: dict[str, IngestContract],
    sources: dict[str, DataSource],
) -> list[str]:
    """Run cross-contract validation rules across a loaded corpus.

    Returns a list of aggregated error messages; empty on success.
    """
    errors: list[str] = []
    for name, source in sources.items():
        if source.ingest_source is None:
            # Legacy path: no cross-contract check required.
            if isinstance(source.periods, FromIngestTemplate):
                errors.append(
                    f"DataSource '{name}': 'periods.mode: from_ingest' requires "
                    f"'ingest_source' to reference a sibling ingest contract"
                )
            continue

        ingest = ingests.get(source.ingest_source)
        if ingest is None:
            errors.append(
                f"DataSource '{name}': ingest_source '{source.ingest_source}' "
                f"does not match any loaded *.ingest.yaml contract"
            )
            continue

        if isinstance(source.periods, FromIngestTemplate):
            partition_mode = (
                ingest.pattern_config.get("partition", {}).get("mode")
                if isinstance(ingest.pattern_config.get("partition"), dict)
                else None
            )
            if partition_mode != "dynamic":
                errors.append(
                    f"DataSource '{name}': 'periods.mode: from_ingest' requires the "
                    f"linked ingest '{ingest.source_name}' to declare "
                    f"'partition.mode: dynamic' (got '{partition_mode}')"
                )
        else:
            ingest_keys = _ingest_partition_keys(ingest)
            # If the ingest pattern doesn't expose enumerable partition keys
            # at load time (e.g. api_resolver / dynamic), skip static
            # alignment rather than raising -- that's the dynamic-linkage
            # rule's job.
            if ingest_keys is None:
                continue
            for period in source.periods:
                pk = period.partition_key
                if pk is None:
                    continue
                if pk not in ingest_keys:
                    errors.append(
                        f"DataSource '{name}': partition_key '{pk}' has no "
                        f"matching entry in linked ingest '{ingest.source_name}'"
                    )
    return errors


def _ingest_partition_keys(ingest: IngestContract) -> set[str] | None:
    """Return the enumerable partition keys for an ingest contract, or None.

    For ``http_urls`` this is the set of ``partition_key`` values in the
    pattern's ``periods`` list. For patterns whose partitions are only
    discovered at runtime (``api_resolver``) this returns ``None`` so
    the caller can skip static alignment.
    """
    if ingest.pattern == "http_urls":
        raw = ingest.pattern_config.get("periods") or []
        if not isinstance(raw, list):
            return set()
        return {str(p.get("partition_key")) for p in raw if isinstance(p, dict)}
    return None


def _validate_partitioned_sink_guards(contract: DataContract) -> list[str]:
    """Static form of the write-time partition guard rails (#401 item 2).

    ``validate_partition_safety`` rejects destructive write modes on
    partitioned assets that declare no ``partition_column`` -- but only at
    write time. A misconfigured pipeline that dies before reaching
    ``database.write()`` (data-platform PF-4: OOMKilled runs) never trips the
    guard, so the misconfiguration survives indefinitely. The combination is
    statically detectable whenever the contract itself declares the
    partitioning, so reject it at load and pay seconds at import instead of a
    dead pipeline at runtime.

    A contract counts as partitioned when its resolved data source declares
    dynamic partitions (``periods.mode: from_ingest``) or any enumerated
    period carries a ``partition_key``. Sinks whose ``mode`` is left to asset
    metadata are skipped -- the write-time guard remains the backstop there.
    """
    data_source = contract.data_source
    if data_source is None:
        return []

    periods = data_source.periods
    partitioned = isinstance(periods, FromIngestTemplate) or (
        isinstance(periods, list) and any(p.partition_key is not None for p in periods)
    )
    if not partitioned:
        return []

    errors: list[str] = []
    for i, sink in enumerate(contract.sinks):
        if not isinstance(sink, dict) or sink.get("type") != "table":
            continue
        mode = sink.get("mode")
        if mode in ("full_refresh", "scd2") and not sink.get("partition_column"):
            errors.append(
                f"Sink {i}: mode '{mode}' with no partition_column, but data "
                f"source '{data_source.source_name}' is partitioned. Without "
                f"partition_column, every partitioned run would destroy data "
                f"from other partitions; set partition_column to scope the "
                f"write to the active partition"
            )
    return errors


def load_contract(path: str | Path) -> DataContract:
    """Load and validate a contract from a YAML file.

    Args:
        path: Path to the contract YAML file

    Returns:
        Parsed and validated DataContract

    Raises:
        ContractNotFoundError: If file doesn't exist
        ContractValidationError: If YAML structure is invalid
    """
    path = Path(path)

    if not path.exists():
        raise ContractNotFoundError(f"Contract file not found: {path}")

    try:
        with path.open() as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ContractValidationError(f"Invalid YAML in {path}: {e}") from e

    if data is None:
        raise ContractValidationError(f"Empty contract file: {path}")

    errors = validate_contract_schema(data)
    if errors:
        raise ContractValidationError(
            f"Contract validation failed for {path}:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    contract = _parse_contract(data)

    # Resolve data_source reference if present
    ds_path_str = data.get("data_source")
    if ds_path_str is not None:
        ds_path = Path(ds_path_str)
        if not ds_path.is_absolute():
            ds_path = path.parent / ds_path
        contract.data_source = load_data_source(ds_path)

        # Static partition guard rails need the resolved data source (the
        # partitioning declaration lives in *.source.yaml), so they run here
        # rather than in validate_contract_schema (#401 item 2).
        guard_errors = _validate_partitioned_sink_guards(contract)
        if guard_errors:
            raise ContractValidationError(
                f"Contract validation failed for {path}:\n"
                + "\n".join(f"  - {e}" for e in guard_errors)
            )

    # Warn about columns that lack an explicit pii annotation.
    # Unannotated columns default to pii=True (safe by default), but the warning
    # nudges engineers to explicitly classify every column.
    raw_columns = data.get("schema", {}).get("columns", [])
    unannotated = [
        c["name"]
        for c in raw_columns
        if isinstance(c, dict) and not c.get("managed", False) and "pii" not in c
    ]
    if unannotated:
        logger.warning(
            "Contract '%s': columns %s have no explicit 'pii' annotation. "
            "They default to pii=true (treated as PII). Add 'pii: false' to "
            "opt out or 'pii: true' to suppress this warning.",
            contract.asset,
            unannotated,
        )

    # Migration 019 (#308) Phase 3: populate stable content fingerprints
    # AFTER all parsing + data_source resolution so the hash covers the
    # final shape of the contract, not a half-parsed intermediate.
    from moncpipelib.contracts.hashing import (
        compute_contract_hash,
        compute_schema_fingerprint,
    )

    contract.contract_hash = compute_contract_hash(contract)
    contract.schema_fingerprint = compute_schema_fingerprint(contract)

    return contract


def _sink_identities(data: dict[str, Any]) -> frozenset[str]:
    """Extract sink-qualified identities (``"schema/table"``) from raw contract YAML.

    Only table-type sinks that declare both ``schema`` and ``table`` contribute
    an identity; sinks missing either field cannot disambiguate contracts.
    """
    sinks = data.get("sinks")
    if not isinstance(sinks, list):
        return frozenset()
    ids: set[str] = set()
    for sink in sinks:
        if not isinstance(sink, dict) or sink.get("type") != "table":
            continue
        table = sink.get("table")
        schema = sink.get("schema")
        if table and schema:
            ids.add(f"{schema}/{table}")
    return frozenset(ids)


def _build_contract_index(
    search_paths: list[Path | str],
) -> _ContractIndex:
    """Build a layer- and sink-aware contract index by scanning search paths.

    Recursively discovers ``*.contract.yaml`` files under each search path and
    reads the ``asset``, ``layer``, and ``sinks`` fields from each.  Entries
    are indexed both by asset name and by sink-qualified identity
    (``"schema/table"``), allowing contracts with the same asset name but
    different layers OR different sinks to coexist (#405).

    A same-asset/same-layer duplicate that cannot be disambiguated by sink
    (overlapping sink identities, or either contract lacking one) is handled
    by origin:

    - **Same search path**: raises ``ContractValidationError`` at index
      build.  The pre-#405 behavior -- warn and silently drop the second
      contract -- combined with the last-component lookup fallback to
      resolve cross-schema writes to the WRONG contract with no signal.
    - **Different search paths**: the contract from the earlier search path
      wins and the later one is shadowed with a warning, preserving the
      documented search-path priority order (override semantics).

    The result is cached per unique set of resolved paths so that repeated
    calls (one per asset write) only scan the filesystem once.

    Args:
        search_paths: Directories to scan recursively.

    Returns:
        A ``_ContractIndex`` with ``by_asset`` and ``by_sink`` maps.
    """
    cache_key = frozenset(str(Path(p).resolve()) for p in search_paths)
    if cache_key in _contract_index_cache:
        return _contract_index_cache[cache_key]

    by_asset: dict[str, list[_IndexedContract]] = {}
    by_sink: dict[str, list[_IndexedContract]] = {}
    seen_files: set[Path] = set()
    for search_dir in search_paths:
        search_dir = Path(search_dir)
        if not search_dir.is_dir():
            continue
        search_root = str(search_dir.resolve())
        for contract_file in sorted(search_dir.rglob(CONTRACT_FILE_PATTERN)):
            # Overlapping search paths (e.g. a parent and its child dir) can
            # discover the same file twice -- it is one contract, not a
            # duplicate.
            resolved_file = contract_file.resolve()
            if resolved_file in seen_files:
                continue
            seen_files.add(resolved_file)

            try:
                with contract_file.open() as f:
                    data = yaml.safe_load(f)
            except (yaml.YAMLError, OSError):
                logger.warning("Failed to read contract file: %s", contract_file)
                continue

            if not isinstance(data, dict) or "asset" not in data:
                continue

            asset_name = str(data["asset"])
            contract_layer: str | None = data.get("layer")

            # Validate layer value against VALID_LAYERS
            if contract_layer is not None and contract_layer not in VALID_LAYERS:
                raise ContractValidationError(
                    f"Invalid layer '{contract_layer}' in contract '{contract_file}'. "
                    f"Valid layers are: {sorted(VALID_LAYERS)}"
                )

            entry = _IndexedContract(
                layer=contract_layer,
                path=contract_file,
                sink_ids=_sink_identities(data),
                search_root=search_root,
            )

            entries = by_asset.setdefault(asset_name, [])
            shadowed = False
            for existing in entries:
                if existing.layer != contract_layer:
                    continue
                # Same asset AND same layer: sink identity can disambiguate
                # (distinct physical tables -- both contracts coexist).
                if (
                    entry.sink_ids
                    and existing.sink_ids
                    and entry.sink_ids.isdisjoint(existing.sink_ids)
                ):
                    continue
                if existing.search_root != search_root:
                    # Search paths are checked in priority order: an earlier
                    # path's contract shadows a later path's duplicate.
                    logger.warning(
                        "Contract for asset '%s' (layer=%s) at %s is shadowed "
                        "by %s from an earlier search path",
                        asset_name,
                        contract_layer,
                        contract_file,
                        existing.path,
                    )
                    shadowed = True
                    break
                raise ContractValidationError(
                    f"Duplicate contract for asset '{asset_name}' "
                    f"(layer={contract_layer}): '{contract_file}' and "
                    f"'{existing.path}' cannot be disambiguated by sink. "
                    f"Give each contract a distinct table sink "
                    f"(schema + table) or a distinct layer."
                )
            if shadowed:
                continue
            entries.append(entry)

            for sink_id in entry.sink_ids:
                by_sink.setdefault(sink_id, []).append(entry)

    index = _ContractIndex(by_asset=by_asset, by_sink=by_sink)
    _contract_index_cache[cache_key] = index
    return index


def _resolve_from_index(
    index: _ContractIndex,
    asset_name: str,
    layer: str | None,
) -> Path | None:
    """Resolve a contract from the layer- and sink-aware index.

    Uses a three-stage strategy:

    1. **Exact asset match** on the full ``asset_name`` against the
       contracts' ``asset`` field.
    2. **Sink-qualified match** -- if ``asset_name`` contains ``/``
       (e.g., from Dagster's ``AssetKey.to_user_string()``), match it
       against sink identities (``"schema/table"``) declared by contract
       table sinks (#405).  Repo-convention asset keys of
       ``[schema, table]`` resolve deterministically here even when
       several contracts share a bare ``asset`` name.  Keys with extra
       leading components are also tried on their last two components.
    3. **Last-component fallback** (legacy) -- match the portion after
       the last ``/`` against the ``asset`` field.

    When multiple contracts survive a stage, the ``layer`` parameter
    disambiguates.  If disambiguation fails, a ``ContractValidationError``
    is raised -- never a silent first-match (#405).
    """
    # Stage 1: exact match on the contract 'asset' field
    entries = index.by_asset.get(asset_name)

    # Stage 2: sink-qualified identity ("schema/table")
    if entries is None and "/" in asset_name:
        entries = index.by_sink.get(asset_name)
        if entries is None:
            components = asset_name.split("/")
            if len(components) > 2:
                entries = index.by_sink.get("/".join(components[-2:]))

    # Stage 3: legacy last-component fallback on the 'asset' field
    if entries is None and "/" in asset_name:
        entries = index.by_asset.get(asset_name.rsplit("/", 1)[-1])

    if not entries:
        return None

    # Single match -- return immediately
    if len(entries) == 1:
        return entries[0].path

    # Multiple matches -- disambiguate by layer
    layers_found = sorted(str(e.layer) for e in entries)
    files_found = sorted(str(e.path) for e in entries)

    if layer is None:
        raise ContractValidationError(
            f"Multiple contracts found for asset '{asset_name}' "
            f"(layers: {layers_found}, files: {files_found}) but no layer "
            f"was provided. Pass layer= to disambiguate, or look the asset "
            f"up by its sink-qualified name ('schema/table'). "
            f"Valid layers are: {sorted(VALID_LAYERS)}"
        )

    layer_matches = [e.path for e in entries if e.layer == layer]
    if len(layer_matches) == 1:
        return layer_matches[0]

    raise ContractValidationError(
        f"Multiple contracts found for asset '{asset_name}' "
        f"(layers: {layers_found}, files: {files_found}) and layer='{layer}' "
        f"did not resolve to a unique match. Look the asset up by its "
        f"sink-qualified name ('schema/table') to disambiguate same-layer "
        f"contracts. Valid layers are: {sorted(VALID_LAYERS)}"
    )


def _clear_contract_index_cache() -> None:
    """Clear the contract index cache.  Intended for testing."""
    _contract_index_cache.clear()


def load_contract_for_asset(
    asset_name: str,
    layer: str | None = None,
    search_paths: list[Path | str] | None = None,
    caller_file: str | None = None,
) -> DataContract | None:
    """Find and load contract for an asset by convention.

    When ``search_paths`` is provided (**recommended**), recursively discovers
    ``*.contract.yaml`` files and matches by the ``asset`` field inside the
    YAML -- not by filename.  This supports any naming convention for contract
    files and nested directory layouts (e.g. Dagster ``defs/`` folders).

    Fallback strategies (deprecated):

    1. Directory of the calling file (auto-discovered via stack frame)
    2. Current working directory
    3. ``assets/{layer}/`` directory

    .. deprecated::
        Strategies 1-3 use filename-based matching (``{asset_name}.contract.yaml``)
        and are deprecated.  They will be removed in a future version.  Pass
        explicit ``search_paths`` to ensure deterministic contract resolution
        across all environments (tests, CI, K8s).

    Args:
        asset_name: Name of the asset (matched against the contract's ``asset`` field).
            If the name contains ``/`` (e.g., from Dagster's
            ``AssetKey.to_user_string()``), the loader also tries matching it
            against sink-qualified identities (``"schema/table"`` from contract
            table sinks, #405) and, as a legacy fallback, the last component
            after ``/``.
        layer: Data layer (bronze/silver/gold), optional.  Used to disambiguate
            when multiple contracts share the same asset name but have different
            layers.  Also used by deprecated fallback strategies.
        search_paths: Directories to search recursively (**recommended**).
        caller_file: Path to the file calling this function (auto-detected if None).
            Only used by deprecated fallback strategies.

    Returns:
        DataContract if found, None otherwise.
    """
    # --- Strategy 1: explicit search_paths with content-based matching (RECOMMENDED) ---
    if search_paths:
        index = _build_contract_index(search_paths)
        contract_path = _resolve_from_index(index, asset_name, layer)
        if contract_path is not None:
            return load_contract(contract_path)

        # Contract not found -- emit a diagnostic warning when contracts DO
        # exist in the search path but none match.  This catches mismatches
        # between the Dagster asset name and the contract's ``asset`` field.
        if index.by_asset:
            available = sorted(set(index.by_asset) | set(index.by_sink))
            close = difflib.get_close_matches(asset_name, available, n=3, cutoff=0.4)
            msg = (
                f"No contract found for asset '{asset_name}'. "
                f"{len(index.by_asset)} contract(s) discovered in search paths but none "
                f"matched. Available asset names in contracts: {available}"
            )
            if close:
                msg += f". Close matches: {close}"
            logger.warning(msg)

        # When search_paths is explicitly provided, don't fall through to
        # deprecated strategies -- the contract simply doesn't exist.
        return None

    # --- All strategies below are DEPRECATED (filename-based matching) ---
    contract_filename = f"{asset_name}.contract.yaml"

    # Stack-frame auto-discovery (DEPRECATED)
    if caller_file is None:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_file = frame.f_back.f_globals.get("__file__")

    if caller_file:
        caller_dir = Path(caller_file).parent
        contract_path = caller_dir / contract_filename
        if contract_path.exists():
            warnings.warn(
                "load_contract_for_asset() resolved contract via stack-frame "
                "auto-discovery. This is deprecated and will be removed in a "
                "future version. Pass explicit search_paths instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return load_contract(contract_path)

    # Current working directory fallback (DEPRECATED)
    cwd_path = Path.cwd() / contract_filename
    if cwd_path.exists():
        warnings.warn(
            "load_contract_for_asset() resolved contract via current working "
            "directory fallback. This is deprecated and will be removed in a "
            "future version. Pass explicit search_paths instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return load_contract(cwd_path)

    # assets/{layer}/ directory fallback (DEPRECATED)
    if layer:
        layer_path = Path.cwd() / "assets" / layer / contract_filename
        if layer_path.exists():
            warnings.warn(
                f"load_contract_for_asset() resolved contract via assets/{layer}/ "
                "directory fallback. This is deprecated and will be removed in a "
                "future version. Pass explicit search_paths instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return load_contract(layer_path)

    return None


def validate_contract_schema(data: dict[str, Any]) -> list[str]:
    """Validate contract YAML structure.

    Args:
        data: Parsed YAML dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: list[str] = []

    # Check required fields
    missing_fields = REQUIRED_FIELDS - set(data.keys())
    if missing_fields:
        errors.append(f"Missing required fields: {', '.join(sorted(missing_fields))}")

    # Check for unknown top-level fields
    errors.extend(_check_unknown_keys(data, KNOWN_TOP_LEVEL_FIELDS, "Contract"))

    # Validate pipeline_id is a valid UUID
    pipeline_id = data.get("pipeline_id")
    if pipeline_id is not None:
        try:
            _uuid_mod.UUID(str(pipeline_id))
        except ValueError:
            errors.append(f"'pipeline_id' must be a valid UUID, got '{pipeline_id}'")

    # Check version
    version = data.get("version")
    if version and version not in SUPPORTED_VERSIONS:
        errors.append(f"Unsupported contract version '{version}'. Supported: {SUPPORTED_VERSIONS}")

    # Validate schema structure
    schema = data.get("schema")
    if schema:
        if not isinstance(schema, dict):
            errors.append("'schema' must be an object")
        else:
            errors.extend(_check_unknown_keys(schema, KNOWN_SCHEMA_FIELDS, "'schema'"))
            columns = schema.get("columns")
            if not columns:
                errors.append("'schema.columns' is required and must not be empty")
            elif not isinstance(columns, list):
                errors.append("'schema.columns' must be a list")
            else:
                for i, col in enumerate(columns):
                    col_errors = _validate_column(col, i)
                    errors.extend(col_errors)

    # Validate expectations if present
    expectations = data.get("expectations")
    if expectations:
        if not isinstance(expectations, list):
            errors.append("'expectations' must be a list")
        else:
            for i, exp in enumerate(expectations):
                exp_errors = _validate_expectation(exp, i)
                errors.extend(exp_errors)

    # Validate owner if present
    owner = data.get("owner")
    if owner:
        if not isinstance(owner, dict):
            errors.append("'owner' must be an object")
        else:
            if "team" not in owner:
                errors.append("'owner.team' is required")
            errors.extend(_check_unknown_keys(owner, KNOWN_OWNER_FIELDS, "'owner'"))

    # Validate upstream if present
    upstream = data.get("upstream")
    if upstream:
        if not isinstance(upstream, list):
            errors.append("'upstream' must be a list")
        else:
            for i, up in enumerate(upstream):
                if not isinstance(up, dict):
                    errors.append(f"Upstream {i}: must be an object")
                else:
                    errors.extend(_check_unknown_keys(up, KNOWN_UPSTREAM_FIELDS, f"Upstream {i}"))

    # Validate sla if present
    sla = data.get("sla")
    if sla:
        if not isinstance(sla, dict):
            errors.append("'sla' must be an object")
        else:
            errors.extend(_check_unknown_keys(sla, KNOWN_SLA_FIELDS, "'sla'"))

    # Validate testing if present
    testing = data.get("testing")
    if testing:
        if not isinstance(testing, dict):
            errors.append("'testing' must be an object")
        else:
            errors.extend(_check_unknown_keys(testing, KNOWN_TESTING_FIELDS, "'testing'"))

    # Validate lineage if present
    lineage = data.get("lineage")
    if lineage:
        if not isinstance(lineage, dict):
            errors.append("'lineage' must be an object")
        else:
            errors.extend(_check_unknown_keys(lineage, KNOWN_LINEAGE_FIELDS, "'lineage'"))

    # Validate tags if present
    tags = data.get("tags")
    if tags is not None:
        if not isinstance(tags, dict):
            errors.append("'tags' must be an object with string keys and string values")
        else:
            for k, v in tags.items():
                if not isinstance(k, str):
                    errors.append(f"'tags' key must be a string, got {type(k).__name__}")
                if not isinstance(v, str):
                    errors.append(f"'tags[\"{k}\"]' value must be a string, got {type(v).__name__}")

    # Validate sources if present
    sources_raw = data.get("sources")
    if sources_raw:
        if not isinstance(sources_raw, list):
            errors.append("'sources' must be a list")
        else:
            for i, source in enumerate(sources_raw):
                errors.extend(
                    _validate_source_sink_entry(source, i, "Source", KNOWN_SOURCE_TABLE_FIELDS)
                )

    # Validate sinks if present
    sinks_raw = data.get("sinks")
    if sinks_raw:
        if not isinstance(sinks_raw, list):
            errors.append("'sinks' must be a list")
        else:
            for i, sink in enumerate(sinks_raw):
                errors.extend(_validate_source_sink_entry(sink, i, "Sink", KNOWN_SINK_TABLE_FIELDS))

    # Cross-validate sink column references against schema columns.
    # partition_column is deliberately NOT cross-referenced here: it is
    # injected into the DataFrame at write time (_inject_period_partition_column)
    # and is conventionally absent from the contract schema block.
    schema_data = data.get("schema")
    if schema_data and isinstance(schema_data, dict):
        schema_columns_raw = schema_data.get("columns", [])
        if isinstance(schema_columns_raw, list):
            known_col_names = {
                str(c.get("name", "")) for c in schema_columns_raw if isinstance(c, dict)
            }
            nullable_col_names = {
                str(c.get("name", ""))
                for c in schema_columns_raw
                if isinstance(c, dict) and c.get("nullable") is True
            }
            schema_pk_cols = [
                str(c.get("name", ""))
                for c in schema_columns_raw
                if isinstance(c, dict) and c.get("primary_key") is True
            ]
            sinks_for_xref = data.get("sinks", [])
            if isinstance(sinks_for_xref, list):
                for i, sink in enumerate(sinks_for_xref):
                    if not isinstance(sink, dict) or sink.get("type") != "table":
                        continue
                    sink_partition_column = sink.get("partition_column")
                    for ref_field in ("business_key", "tracked_columns", "primary_key"):
                        ref = sink.get(ref_field)
                        if ref is None:
                            continue
                        cols = [ref] if isinstance(ref, str) else list(ref)
                        for col in cols:
                            if not isinstance(col, str) or col in known_col_names:
                                continue
                            # The partition column is injected into the DataFrame
                            # at write time and is conventionally absent from the
                            # schema block; guard 2 below *requires* it in an
                            # upsert primary_key, so it cannot be a ghost ref.
                            if ref_field == "primary_key" and col == sink_partition_column:
                                continue
                            errors.append(
                                f"Sink {i}: '{ref_field}' references column '{col}' "
                                f"which is not defined in the schema"
                            )

                    if sink.get("mode") != "upsert":
                        continue

                    # Effective upsert conflict key: the sink-level primary_key
                    # when declared, else the schema's primary_key: true columns
                    # (mirrors ContractReconciler.reconcile_primary_key).
                    sink_pk_raw = sink.get("primary_key")
                    if isinstance(sink_pk_raw, str):
                        effective_pk = [sink_pk_raw]
                    elif isinstance(sink_pk_raw, list):
                        effective_pk = [c for c in sink_pk_raw if isinstance(c, str)]
                    else:
                        effective_pk = schema_pk_cols

                    # #401 item 3 (data-platform dim_hcpcs): NULLs never match
                    # ON CONFLICT, so a nullable conflict-key member silently
                    # duplicates NULL-keyed rows on every re-materialization.
                    nullable_pk = [c for c in effective_pk if c in nullable_col_names]
                    if nullable_pk:
                        errors.append(
                            f"Sink {i}: mode 'upsert' primary_key member(s) "
                            f"{nullable_pk} are declared nullable in the schema. "
                            f"NULL keys bypass ON CONFLICT matching and duplicate "
                            f"silently; make them non-nullable or choose a "
                            f"different primary_key"
                        )

                    # #401 item 2, guard 2 (static form): an upsert scoped by
                    # partition_column whose conflict key does not include that
                    # column would match records across partitions at runtime.
                    partition_column = sink.get("partition_column")
                    if (
                        isinstance(partition_column, str)
                        and partition_column
                        and effective_pk
                        and partition_column not in effective_pk
                    ):
                        errors.append(
                            f"Sink {i}: mode 'upsert' declares partition_column "
                            f"'{partition_column}' but primary_key {effective_pk} "
                            f"does not include it. The upsert would match records "
                            f"across partitions; add '{partition_column}' to "
                            f"primary_key or remove partition_column"
                        )

    # Reject inline periods (now defined in *.source.yaml files)
    if "periods" in data:
        errors.append(
            "Contract contains 'periods' which is no longer supported inline. "
            "Periods are now defined in dedicated *.source.yaml files and "
            "referenced via the 'data_source' field on the contract."
        )

    # Validate data_source if present (must be a string path)
    data_source = data.get("data_source")
    if data_source is not None and not isinstance(data_source, str):
        errors.append("'data_source' must be a string path to a *.source.yaml file")

    return errors


def _validate_column(col: Any, index: int) -> list[str]:
    """Validate a column definition."""
    errors: list[str] = []
    prefix = f"Column {index}"

    if not isinstance(col, dict):
        return [f"{prefix}: must be an object"]

    # Check required column fields
    name = col.get("name")
    if not name:
        errors.append(f"{prefix}: 'name' is required")
    else:
        prefix = f"Column '{name}'"

    col_type = col.get("type")
    if not col_type:
        errors.append(f"{prefix}: 'type' is required")
    else:
        valid_types = {t.value for t in ColumnType}
        if col_type not in valid_types:
            errors.append(f"{prefix}: invalid type '{col_type}'. Valid types: {valid_types}")

    if "nullable" not in col:
        errors.append(f"{prefix}: 'nullable' is required")
    elif not isinstance(col.get("nullable"), bool):
        errors.append(f"{prefix}: 'nullable' must be a boolean")
    elif col.get("nullable") and col.get("primary_key") is True:
        # A nullable primary-key member can never be backed by a real
        # PRIMARY KEY constraint, and under upsert its NULLs bypass
        # ON CONFLICT entirely (rows silently duplicate on every run --
        # #401 item 3). Contradictory by construction, so reject at load.
        errors.append(
            f"{prefix}: 'primary_key: true' cannot be combined with "
            f"'nullable: true'. Primary-key columns must be non-nullable; "
            f"NULL keys bypass upsert ON CONFLICT matching and duplicate "
            f"silently."
        )

    # Validate pii if present
    pii = col.get("pii")
    if pii is not None and not isinstance(pii, bool):
        errors.append(f"{prefix}: 'pii' must be a boolean")

    # Validate phi if present
    phi = col.get("phi")
    if phi is not None and not isinstance(phi, bool):
        errors.append(f"{prefix}: 'phi' must be a boolean")

    # Check for unknown column fields
    errors.extend(_check_unknown_keys(col, KNOWN_COLUMN_FIELDS, prefix))

    # Validate tests if present
    tests = col.get("tests")
    if tests:
        if not isinstance(tests, list):
            errors.append(f"{prefix}: 'tests' must be a list")
        else:
            for i, test in enumerate(tests):
                test_errors = _validate_test(test, f"{prefix} test {i}")
                errors.extend(test_errors)

    return errors


def _validate_test(test: Any, prefix: str) -> list[str]:
    """Validate a column test definition."""
    errors: list[str] = []

    # Tests can be simple strings or dicts
    if isinstance(test, str):
        # Simple test like "not_null" or "unique"
        if test not in SIMPLE_TESTS:
            errors.append(f"{prefix}: unknown simple test '{test}'")
        return errors

    if not isinstance(test, dict):
        errors.append(f"{prefix}: must be a string or object")
        return errors

    # Dict test with parameters
    # Extract test type from first non-modifier key
    test_keys = [k for k in test if k not in TEST_MODIFIERS]

    if not test_keys:
        errors.append(f"{prefix}: no test type specified")
    elif len(test_keys) > 1:
        errors.append(f"{prefix}: multiple test types specified: {test_keys}")
    else:
        test_type = str(test_keys[0])
        if test_type not in KNOWN_COLUMN_TEST_TYPES:
            errors.append(
                f"{prefix}: unknown test type '{test_type}'.{_suggest(test_type, KNOWN_COLUMN_TEST_TYPES)}"
            )
        else:
            errors.extend(_validate_test_params(test[test_keys[0]], test_type, prefix))

    # Validate severity if present
    severity = test.get("severity")
    if severity and severity not in {"error", "warn"}:
        errors.append(f"{prefix}: invalid severity '{severity}'. Must be 'error' or 'warn'")

    # Validate when condition if present
    when = test.get("when")
    if when and when not in {"not_null"}:
        errors.append(f"{prefix}: invalid 'when' condition '{when}'. Must be 'not_null'")

    return errors


def _validate_test_params(value: Any, test_type: str, prefix: str) -> list[str]:
    """Validate the parameter mapping nested under a dict-form column test.

    ``_parse_test`` copies the nested mapping verbatim into
    ``ColumnTest.parameters`` and reads ``severity``/``when`` only from the
    sibling level, so a reserved or misspelled key nested here is silently
    ignored at runtime (#394) -- reject it at load time instead.
    """
    errors: list[str] = []
    if not isinstance(value, dict):
        # Scalar shorthand (e.g. ``greater_than: 0``) parses to {"value": ...}
        return errors
    known = KNOWN_COLUMN_TEST_PARAMS[test_type]
    for key in sorted(str(k) for k in value):
        if key in known:
            continue
        if key in TEST_MODIFIERS:
            errors.append(
                f"{prefix}: '{key}' must be a sibling of '{test_type}', not nested "
                f"inside its parameters -- as written it would be silently ignored. "
                f"Move it up one level."
            )
        else:
            valid = ", ".join(sorted(known)) or "none"
            errors.append(
                f"{prefix}: unknown parameter '{key}' for '{test_type}'. "
                f"Valid parameters: {valid}.{_suggest(key, known)}"
            )
    return errors


def _validate_expectation_params(value: Any, exp_type: str, prefix: str) -> list[str]:
    """Validate the parameter mapping nested under a table expectation type.

    ``_parse_expectation`` copies the nested mapping verbatim into
    ``TableExpectation.parameters`` and reads ``severity`` only from the
    sibling level, so a nested ``severity: warn`` silently enforces at
    ``error`` (#394) -- reject misplaced and unknown keys at load time.
    """
    errors: list[str] = []
    known = KNOWN_EXPECTATION_PARAMS[exp_type]
    if value is None:
        return errors
    if not isinstance(value, dict):
        expected = ", ".join(sorted(known)) or "no parameters"
        errors.append(
            f"{prefix}: '{exp_type}' takes a mapping of parameters ({expected}), "
            f"not a bare value -- a bare value would be silently ignored"
        )
        return errors
    for key in sorted(str(k) for k in value):
        if key in known:
            continue
        if key == "severity":
            errors.append(
                f"{prefix}: 'severity' must be a sibling of '{exp_type}', not nested "
                f"inside its parameters -- as written it would be silently ignored "
                f"and the expectation would enforce at 'error'. Move it up one level."
            )
        elif key == "when":
            errors.append(
                f"{prefix}: 'when' is not supported on table expectations "
                f"(only on column tests) and would be silently ignored"
            )
        else:
            valid = ", ".join(sorted(known)) or "none"
            errors.append(
                f"{prefix}: unknown parameter '{key}' for '{exp_type}'. "
                f"Valid parameters: {valid}.{_suggest(key, known)}"
            )
    return errors


def _validate_expectation(exp: Any, index: int) -> list[str]:
    """Validate a table expectation definition."""
    errors: list[str] = []
    prefix = f"Expectation {index}"

    if not isinstance(exp, dict):
        return [f"{prefix}: must be an object"]

    # Extract expectation type from first non-severity key
    exp_keys = [k for k in exp if k != "severity"]

    if not exp_keys:
        errors.append(f"{prefix}: no expectation type specified")
    elif len(exp_keys) > 1:
        errors.append(f"{prefix}: multiple expectation types specified: {exp_keys}")
    else:
        exp_type = str(exp_keys[0])
        if exp_type not in KNOWN_EXPECTATION_TYPES:
            errors.append(
                f"{prefix}: unknown expectation type '{exp_type}'.{_suggest(exp_type, KNOWN_EXPECTATION_TYPES)}"
            )
        else:
            errors.extend(_validate_expectation_params(exp[exp_keys[0]], exp_type, prefix))

    # Validate severity if present
    severity = exp.get("severity")
    if severity and severity not in {"error", "warn"}:
        errors.append(f"{prefix}: invalid severity '{severity}'. Must be 'error' or 'warn'")

    return errors


def _parse_contract(data: dict[str, Any]) -> DataContract:
    """Parse validated YAML data into a DataContract object."""
    # Parse schema
    schema_data = data["schema"]
    columns = [_parse_column(col) for col in schema_data["columns"]]
    schema = Schema(columns=columns, strict=schema_data.get("strict", True))

    # Parse optional owner
    owner = None
    owner_data = data.get("owner")
    if owner_data:
        owner = Owner(
            team=owner_data["team"],
            contact=owner_data.get("contact"),
            slack_channel=owner_data.get("slack_channel"),
        )

    # Parse optional expectations
    expectations = []
    exp_data = data.get("expectations", [])
    for exp in exp_data:
        expectations.append(_parse_expectation(exp))

    # Parse optional upstream
    upstream = []
    upstream_data = data.get("upstream", [])
    for up in upstream_data:
        upstream.append(
            UpstreamDependency(
                name=up["name"],
                type=up["type"],
                system=up.get("system"),
                description=up.get("description"),
            )
        )

    # Parse optional SLA
    sla = None
    sla_data = data.get("sla")
    if sla_data:
        sla = SLA(
            freshness_hours=sla_data.get("freshness_hours"),
            update_frequency=sla_data.get("update_frequency"),
            availability_percent=sla_data.get("availability_percent"),
        )

    # Parse optional sources
    sources: list[dict[str, Any]] = data.get("sources", [])

    # Parse optional sinks
    sinks: list[dict[str, Any]] = data.get("sinks", [])

    # Parse optional testing configuration
    testing = None
    testing_data = data.get("testing")
    if testing_data:
        testing = TestingConfig(
            enabled=testing_data.get("enabled", True),
            source_row_limit=testing_data.get("source_row_limit", 1000),
            source_where_clause=testing_data.get("source_where_clause"),
            expected_min_rows=testing_data.get("expected_min_rows"),
            expected_max_rows=testing_data.get("expected_max_rows"),
            timeout_seconds=testing_data.get("timeout_seconds", 300),
        )

    # Parse optional lineage configuration
    lineage = None
    lineage_data = data.get("lineage")
    if lineage_data:
        lineage = LineageConfig(
            enabled=lineage_data.get("enabled", True),
            source_system=lineage_data.get("source_system"),
            transformation_type=lineage_data.get("transformation_type"),
        )

    return DataContract(
        version=data["version"],
        pipeline_id=str(data["pipeline_id"]),
        asset=data["asset"],
        layer=data["layer"],
        schema=schema,
        description=data.get("description"),
        owner=owner,
        expectations=expectations,
        upstream=upstream,
        sla=sla,
        sources=sources,
        sinks=sinks,
        testing=testing,
        lineage=lineage,
        tags=data.get("tags") or {},
        parameters=data.get("parameters") or {},
        data_source=None,  # resolved after construction if data_source path is present
    )


def _parse_periods(
    raw: list[dict[str, Any]] | dict[str, Any] | None,
) -> list[Period] | FromIngestTemplate:
    """Parse a ``periods`` block from YAML.

    Returns either an enumerated ``list[Period]`` (existing shape) or a
    ``FromIngestTemplate`` when the YAML declares
    ``periods: {mode: from_ingest, template: {...}}``. Schema validation
    has already run by the time this is called.
    """
    if isinstance(raw, dict):
        template = raw["template"]
        return FromIngestTemplate(
            source=str(template["source"]),
            effective_from_field=str(template["effective_from_field"]),
        )
    if not raw:
        return []
    periods: list[Period] = []
    for entry in raw:
        eff_from = entry["effective_from"]
        eff_to = entry.get("effective_to")
        # YAML auto-parses dates, but ensure they are date objects
        if isinstance(eff_from, _date):
            eff_from_date = eff_from
        else:
            eff_from_date = _date.fromisoformat(str(eff_from))
        eff_to_date: _date | None = None
        if eff_to is not None:
            eff_to_date = eff_to if isinstance(eff_to, _date) else _date.fromisoformat(str(eff_to))
        pk = entry.get("partition_key")
        periods.append(
            Period(
                source=str(entry["source"]),
                effective_from=eff_from_date,
                effective_to=eff_to_date,
                partition_key=str(pk) if pk is not None else None,
            )
        )
    return periods


def _parse_column(col: dict[str, Any]) -> Column:
    """Parse a column definition from YAML."""
    tests = []
    for test in col.get("tests", []):
        tests.append(_parse_test(test))

    return Column(
        name=col["name"],
        type=ColumnType(col["type"]),
        nullable=col["nullable"],
        description=col.get("description"),
        primary_key=col.get("primary_key", False),
        managed=col.get("managed", False),
        pii=col.get("pii", True),
        # None lets Column.__post_init__ default phi to the pii value (#391)
        phi=col.get("phi"),
        tests=tests,
    )


def _parse_test(test: str | dict[str, Any]) -> ColumnTest:
    """Parse a column test from YAML."""
    if isinstance(test, str):
        # Simple test like "not_null"
        return ColumnTest(test_type=test)

    # Dict test with parameters
    modifiers = {"severity", "when"}
    test_type = None
    parameters: dict[str, Any] = {}

    for key, value in test.items():
        if key in modifiers:
            continue
        test_type = key
        # Handle both "pattern: ^foo$" and "accepted_values: {values: [...]}"
        if isinstance(value, dict):
            parameters = value
        elif value is not None:
            # Single value parameter, e.g., "greater_than: 0"
            parameters = {"value": value}

    if test_type is None:
        test_type = "unknown"

    severity = Severity(test.get("severity", "error"))
    when = test.get("when")

    return ColumnTest(
        test_type=test_type,
        parameters=parameters,
        severity=severity,
        when=when,
    )


def _parse_expectation(exp: dict[str, Any]) -> TableExpectation:
    """Parse a table expectation from YAML."""
    exp_type = None
    parameters: dict[str, Any] = {}

    for key, value in exp.items():
        if key == "severity":
            continue
        exp_type = key
        if isinstance(value, dict):
            parameters = value
        elif value is not None:
            parameters = {"value": value}

    if exp_type is None:
        exp_type = "unknown"

    severity = Severity(exp.get("severity", "error"))

    return TableExpectation(
        expectation_type=exp_type,
        parameters=parameters,
        severity=severity,
    )
