"""Tests for the per-partition ingest manifest model.

Covers:

- ``write_to`` / ``read_from`` round-trip equivalence.
- ``manifest_version > KNOWN_MAX_VERSION`` rejection (forward-compat).
- Malformed JSON / missing fields / wrong types -> ``IngestResolutionError``.
- The ``files`` tuple is preserved across the round trip.
- The ``_json_default`` encoder safety net (#233): direct callers writing
  manifests with date / datetime / UUID / Decimal in resolver_config get
  ISO-string encoding rather than a crash.

Byte-stability + streaming-memory acceptance tests live in
``test_manifest_streaming.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from io import BytesIO
from uuid import UUID

import pytest

from moncpipelib.ingest.exceptions import IngestResolutionError
from moncpipelib.ingest.manifest import (
    KNOWN_MAX_VERSION,
    IngestManifest,
    ManifestFileEntry,
)


def _sample_manifest() -> IngestManifest:
    return IngestManifest(
        manifest_version=1,
        source_id="d4f1e8b6-1c4f-4f3e-8a3a-2c4c2c4c2c4c",
        source_name="rxnorm-full-monthly",
        partition_key="2026-04-26",
        materialized_at="2026-04-26T14:22:11Z",
        resolver={"name": "uts_release", "config": {"release_endpoint": "https://x"}},
        fields={"release_date": "2026-04-26", "release_version": "2026AB"},
        files=(
            ManifestFileEntry(
                path="rxnorm/2026-04-26/RXNCONSO.RRF",
                sha256="deadbeef" * 8,
                size_bytes=1024,
            ),
            ManifestFileEntry(
                path="rxnorm/2026-04-26/RXNREL.RRF",
                sha256="cafef00d" * 8,
                size_bytes=2048,
            ),
        ),
    )


def _write_to_bytes(m: IngestManifest) -> bytes:
    """Helper: serialize a manifest via ``write_to`` and return the bytes."""
    buf = BytesIO()
    m.write_to(buf)
    return buf.getvalue()


def _read_from_bytes(data: bytes) -> IngestManifest:
    """Helper: parse a manifest from a bytes payload via ``read_from``."""
    return IngestManifest.read_from(BytesIO(data))


def test_round_trip_equivalence() -> None:
    original = _sample_manifest()
    parsed = _read_from_bytes(_write_to_bytes(original))
    assert parsed == original


def test_known_max_version_default_is_one() -> None:
    """Sanity check: when ``KNOWN_MAX_VERSION`` bumps, the test surface
    that depends on '> 1 is rejected' must be revisited."""
    assert KNOWN_MAX_VERSION == 1


def test_future_manifest_version_is_rejected() -> None:
    text = _write_to_bytes(_sample_manifest()).replace(
        b'"manifest_version": 1', b'"manifest_version": 2'
    )
    with pytest.raises(IngestResolutionError, match="manifest_version' is 2"):
        _read_from_bytes(text)


def test_malformed_json_is_rejected() -> None:
    with pytest.raises(IngestResolutionError, match="not valid JSON"):
        _read_from_bytes(b"{not valid json")


def test_top_level_must_be_object() -> None:
    with pytest.raises(IngestResolutionError, match="must be a JSON object"):
        _read_from_bytes(b'["not", "an", "object"]')


def test_missing_required_field_is_rejected() -> None:
    text = _write_to_bytes(_sample_manifest()).replace(
        b'"source_id": "d4f1e8b6-1c4f-4f3e-8a3a-2c4c2c4c2c4c",\n', b""
    )
    with pytest.raises(IngestResolutionError, match="missing required field"):
        _read_from_bytes(text)


def test_non_int_version_is_rejected() -> None:
    with pytest.raises(IngestResolutionError, match="must be an integer"):
        _read_from_bytes(b'{"manifest_version": "1"}')


def test_bool_version_is_rejected() -> None:
    """``True`` is technically an int in Python, but here it's a type
    error -- guard against it explicitly."""
    with pytest.raises(IngestResolutionError, match="must be an integer"):
        _read_from_bytes(b'{"manifest_version": true}')


def test_files_must_be_a_list() -> None:
    text = (
        b'{"manifest_version": 1, "source_id": "x", "source_name": "y", '
        b'"partition_key": "p", "materialized_at": "t", "resolver": {}, '
        b'"fields": {}, "files": {"not": "a list"}}'
    )
    with pytest.raises(IngestResolutionError, match="'files' must be a JSON array"):
        _read_from_bytes(text)


def test_files_must_be_a_list_not_a_scalar() -> None:
    """Sister assertion to ``test_files_must_be_a_list``: a scalar at the
    ``files`` key (e.g. accidental string) is also rejected, not silently
    coerced to an empty file list."""
    text = (
        b'{"manifest_version": 1, "source_id": "x", "source_name": "y", '
        b'"partition_key": "p", "materialized_at": "t", "resolver": {}, '
        b'"fields": {}, "files": "wrong"}'
    )
    with pytest.raises(IngestResolutionError, match="'files' must be a JSON array"):
        _read_from_bytes(text)


def test_missing_files_key_is_rejected() -> None:
    """A manifest with NO ``files`` key at all must be rejected, not
    silently parsed as ``files=()``.  Streaming through the array means
    ``files`` never lands in ``state``, so without the
    ``saw_files_array`` flag the absence check would pass and we'd
    return an empty-files manifest -- a HIPAA 164.312(c)(1) integrity
    gap (the partition-completeness authority would say "complete"
    when the manifest is structurally malformed).
    """
    text = (
        b'{"manifest_version": 1, "source_id": "x", "source_name": "y", '
        b'"partition_key": "p", "materialized_at": "t", "resolver": {}, '
        b'"fields": {}}'  # no "files" key at all
    )
    with pytest.raises(IngestResolutionError, match="missing required field: 'files'"):
        _read_from_bytes(text)


def test_file_entry_missing_field_is_rejected() -> None:
    text = (
        b'{"manifest_version": 1, "source_id": "x", "source_name": "y", '
        b'"partition_key": "p", "materialized_at": "t", "resolver": {}, '
        b'"fields": {}, "files": [{"path": "x"}]}'
    )
    with pytest.raises(IngestResolutionError, match="'files' entry malformed"):
        _read_from_bytes(text)


def test_resolver_must_be_a_mapping() -> None:
    text = (
        b'{"manifest_version": 1, "source_id": "x", "source_name": "y", '
        b'"partition_key": "p", "materialized_at": "t", "resolver": "uts_release", '
        b'"fields": {}, "files": []}'
    )
    with pytest.raises(IngestResolutionError, match="'resolver' must be a mapping"):
        _read_from_bytes(text)


def test_fields_must_be_a_mapping() -> None:
    text = (
        b'{"manifest_version": 1, "source_id": "x", "source_name": "y", '
        b'"partition_key": "p", "materialized_at": "t", "resolver": {}, '
        b'"fields": [], "files": []}'
    )
    with pytest.raises(IngestResolutionError, match="'fields' must be a mapping"):
        _read_from_bytes(text)


def test_empty_files_list_is_allowed() -> None:
    """A manifest with zero files is structurally valid -- the dispatcher
    enforces 'every file lands successfully' as a runtime contract; the
    model itself does not require non-empty ``files``.  This keeps the
    door open for future patterns that legitimately have nothing to land."""
    text = (
        b'{"manifest_version": 1, "source_id": "x", "source_name": "y", '
        b'"partition_key": "p", "materialized_at": "t", "resolver": {}, '
        b'"fields": {}, "files": []}'
    )
    parsed = _read_from_bytes(text)
    assert parsed.files == ()


# ---------------------------------------------------------------------------
# Encoder safety net (issue #233): direct ``write_to`` callers must not crash
# on date / datetime / UUID / Decimal in resolver config.  The dispatcher's
# ``_coerce_jsonable`` is the primary normalizer; this is the safety net.
# ---------------------------------------------------------------------------


def _manifest_with_resolver_config(config: dict[str, object]) -> IngestManifest:
    return IngestManifest(
        manifest_version=1,
        source_id="d4f1e8b6-1c4f-4f3e-8a3a-2c4c2c4c2c4c",
        source_name="fda-ndc",
        partition_key="2024-01-07",
        materialized_at="2024-01-07T00:00:00Z",
        resolver={"name": "calendar", "config": config},
        fields={},
        files=(),
    )


def test_write_to_handles_date_in_resolver_config() -> None:
    """PyYAML parses bare ISO dates (``2024-01-01``) as ``date`` objects.
    The encoder must serialize them as ISO strings, not crash."""
    m = _manifest_with_resolver_config({"start_date": date(2024, 1, 1)})
    out = json.loads(_write_to_bytes(m))
    assert out["resolver"]["config"]["start_date"] == "2024-01-01"


def test_write_to_handles_datetime_in_resolver_config() -> None:
    m = _manifest_with_resolver_config({"cutover_at": datetime(2024, 1, 1, 12, 30, tzinfo=UTC)})
    out = json.loads(_write_to_bytes(m))
    assert out["resolver"]["config"]["cutover_at"] == "2024-01-01T12:30:00+00:00"


def test_write_to_handles_uuid_in_resolver_config() -> None:
    m = _manifest_with_resolver_config(
        {"upstream_id": UUID("12345678-1234-5678-1234-567812345678")}
    )
    out = json.loads(_write_to_bytes(m))
    assert out["resolver"]["config"]["upstream_id"] == "12345678-1234-5678-1234-567812345678"


def test_write_to_handles_decimal_in_resolver_config() -> None:
    m = _manifest_with_resolver_config({"threshold": Decimal("0.95")})
    out = json.loads(_write_to_bytes(m))
    assert out["resolver"]["config"]["threshold"] == "0.95"


def test_write_to_rejects_unsupported_type() -> None:
    """The encoder is intentionally narrow.  An unsupported type should
    surface as a ``TypeError`` rather than be silently coerced -- that
    would mask authoring errors in resolver config."""
    m = _manifest_with_resolver_config({"weird": object()})
    with pytest.raises(TypeError, match="not JSON serializable"):
        _write_to_bytes(m)
