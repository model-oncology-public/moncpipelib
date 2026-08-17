"""Tests for :func:`read_partition_with_manifest`.

Acceptance cases from moncpipelib#350:

- Non-:class:`BlobRef` ref raises :class:`TypeError`.
- Missing required field raises :class:`ManifestFieldError`
  (subclass of :class:`IngestResolutionError`) with the manifest path and
  the sorted available-fields list.
- A required field with value ``""`` raises; a required field with value
  ``0`` does not (explicit None/empty-string check, not ``if not value:``).
- A multi-level glob (``subdir/*.csv``) resolves the manifest from
  ``{prefix}/_manifest.json``, NOT ``{prefix}/subdir/_manifest.json``
  (regression test for the rsplit bug).
- ``manifest_version > KNOWN_MAX_VERSION`` raises
  :class:`IngestResolutionError` (forward-compat check is preserved
  end-to-end).
- The context manager closes the blob stream on success and when the
  caller's ``with`` body raises.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from io import BytesIO
from typing import IO, Literal

import pytest

from moncpipelib.contracts.models import (
    ContractCorpus,
    DataSource,
    FromIngestTemplate,
    IngestContract,
    Period,
)
from moncpipelib.ingest import (
    IngestResolutionError,
    ManifestFieldError,
    read_partition_with_manifest,
)


class _TrackingStream(BytesIO):
    """``BytesIO`` that records whether ``close()`` was called.

    Used to assert the helper closes the stream on success and on
    caller-raised exceptions.
    """

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.closed_via_helper = False

    def close(self) -> None:
        self.closed_via_helper = True
        super().close()


class FakeBlob:
    """In-memory blob stand-in mirroring the test fake in
    ``test_ingest_resolver.py``.

    ``stream`` returns a :class:`_TrackingStream` so the helper's close
    semantics are observable from tests.
    """

    def __init__(
        self,
        listing: list[str] | None = None,
        contents: dict[str, bytes] | None = None,
    ) -> None:
        self._listing = list(listing or [])
        self._contents = dict(contents or {})
        self.streams_opened: list[_TrackingStream] = []

    def list(self, sensitivity: str, prefix: str) -> list[str]:
        del sensitivity, prefix
        return list(self._listing)

    def iter_list(self, sensitivity: str, prefix: str) -> Iterator[str]:
        del sensitivity, prefix
        return iter(self._listing)

    def exists(self, sensitivity: str, path: str) -> bool:
        del sensitivity
        return path in self._contents

    def download(self, sensitivity: str, path: str) -> bytes:
        del sensitivity
        return self._contents[path]

    def stream(self, sensitivity: str, path: str) -> IO[bytes]:
        del sensitivity
        stream = _TrackingStream(self._contents[path])
        self.streams_opened.append(stream)
        return stream


_VALID_MANIFEST = """{
  "fields": {
    "release_date": "2026-04-26",
    "release_version": "2026AA",
    "release_count": 0
  },
  "files": [
    {"path": "umls/2026AA/meta/MRCONSO.RRF", "sha256": "deadbeef", "size_bytes": 1024}
  ],
  "manifest_version": 1,
  "materialized_at": "2026-04-26T14:22:11Z",
  "partition_key": "2026AA",
  "resolver": {"config": {"release_type": "umls-full-release"}, "name": "uts_release"},
  "source_id": "11111111-1111-1111-1111-111111111111",
  "source_name": "umls-meta"
}"""


def _ingest(
    source_name: str = "umls-meta",
    prefix_template: str = "umls/{partition_key}",
    sensitivity: Literal["public", "confidential", "phi"] = "confidential",
) -> IngestContract:
    return IngestContract(
        source_id="11111111-1111-1111-1111-111111111111",
        source_name=source_name,
        sensitivity=sensitivity,
        pattern="api_resolver",
        prefix_template=prefix_template,
        extract=(),
        strip_extensions=(),
        pattern_config={
            "resolver": "uts_release",
            "resolver_config": {"release_type": "umls-full-release"},
            "credential": {"secret_name": "uts-api-key"},
            "partition": {"mode": "dynamic", "key_from": "release_version"},
        },
    )


def _from_template_source(
    template_source: str = "meta/MRCONSO.RRF",
    effective_from_field: str = "release_date",
    source_name: str = "rxnorm-mrconso",
    ingest_source: str = "umls-meta",
) -> DataSource:
    return DataSource(
        source_id="44444444-4444-4444-4444-444444444444",
        source_name=source_name,
        periods=FromIngestTemplate(
            source=template_source,
            effective_from_field=effective_from_field,
        ),
        ingest_source=ingest_source,
    )


def _corpus(ingest: IngestContract, source: DataSource) -> ContractCorpus:
    return ContractCorpus(
        ingests={ingest.source_name: ingest},
        sources={source.source_name: source},
    )


# ---------------------------------------------------------------------------
# Defensive: non-BlobRef refs (legacy URL sources)
# ---------------------------------------------------------------------------


def test_raises_typeerror_when_resolver_returns_raw_url() -> None:
    """A legacy source (``ingest_source is None``) resolves to
    :class:`~moncpipelib.ingest.RawUrl`; the helper rejects it so the
    caller falls back to the direct-fetch path."""
    source = DataSource(
        source_id="33333333-3333-3333-3333-333333333333",
        source_name="legacy",
        periods=[
            Period(
                source="https://legacy.example.com/file.csv",
                effective_from=date(2024, 1, 1),
                partition_key="2024-01-01",
            )
        ],
        ingest_source=None,
    )
    corpus = ContractCorpus(sources={source.source_name: source})

    with (
        pytest.raises(TypeError, match="BlobRef"),
        read_partition_with_manifest(
            source=source,
            partition_key="2024-01-01",
            corpus=corpus,
            blob=FakeBlob(),  # type: ignore[arg-type]
        ),
    ):
        pass


# ---------------------------------------------------------------------------
# Required-field validation
# ---------------------------------------------------------------------------


def test_missing_required_field_raises_with_available_fields_listed() -> None:
    ingest = _ingest()
    source = _from_template_source()
    corpus = _corpus(ingest, source)
    blob = FakeBlob(
        listing=["umls/2026AA/_manifest.json", "umls/2026AA/meta/MRCONSO.RRF"],
        contents={
            "umls/2026AA/_manifest.json": _VALID_MANIFEST.encode("utf-8"),
            "umls/2026AA/meta/MRCONSO.RRF": b"col1,col2\n",
        },
    )

    with (
        pytest.raises(ManifestFieldError) as excinfo,
        read_partition_with_manifest(
            source=source,
            partition_key="2026AA",
            corpus=corpus,
            blob=blob,  # type: ignore[arg-type]
            required_fields=("release_date", "source_url"),
        ),
    ):
        pass

    msg = str(excinfo.value)
    assert "umls/2026AA/_manifest.json" in msg
    assert "'source_url'" in msg
    # Sorted available-fields list: release_count, release_date, release_version.
    assert "['release_count', 'release_date', 'release_version']" in msg


def test_missing_required_field_is_ingest_resolution_error_subclass() -> None:
    """Existing ``except IngestResolutionError`` blocks must still
    catch the new exception."""
    ingest = _ingest()
    source = _from_template_source()
    corpus = _corpus(ingest, source)
    blob = FakeBlob(
        listing=["umls/2026AA/_manifest.json", "umls/2026AA/meta/MRCONSO.RRF"],
        contents={
            "umls/2026AA/_manifest.json": _VALID_MANIFEST.encode("utf-8"),
            "umls/2026AA/meta/MRCONSO.RRF": b"",
        },
    )

    with (
        pytest.raises(IngestResolutionError),
        read_partition_with_manifest(
            source=source,
            partition_key="2026AA",
            corpus=corpus,
            blob=blob,  # type: ignore[arg-type]
            required_fields=("missing_field",),
        ),
    ):
        pass


def test_empty_string_required_field_raises_but_zero_value_passes() -> None:
    """``""`` and ``None`` are treated as missing; numeric ``0`` is not.

    Manifests realistically carry counts, flags, and other falsy-but-
    valid values.  A blanket ``if not value:`` would reject them.
    """
    ingest = _ingest()
    # Manifest with an empty-string field; the regular manifest already
    # has ``release_count: 0`` which exercises the zero-passes path.
    empty_manifest = _VALID_MANIFEST.replace('"release_date": "2026-04-26"', '"release_date": ""')
    source = _from_template_source()
    corpus = _corpus(ingest, source)
    blob = FakeBlob(
        listing=["umls/2026AA/_manifest.json", "umls/2026AA/meta/MRCONSO.RRF"],
        contents={
            "umls/2026AA/_manifest.json": empty_manifest.encode("utf-8"),
            "umls/2026AA/meta/MRCONSO.RRF": b"",
        },
    )

    # release_date is empty -- raises.
    with (
        pytest.raises(ManifestFieldError, match="release_date"),
        read_partition_with_manifest(
            source=source,
            partition_key="2026AA",
            corpus=corpus,
            blob=blob,  # type: ignore[arg-type]
            required_fields=("release_date",),
        ),
    ):
        pass

    # release_count is 0 -- passes.
    with read_partition_with_manifest(
        source=source,
        partition_key="2026AA",
        corpus=corpus,
        blob=blob,  # type: ignore[arg-type]
        required_fields=("release_count",),
    ) as (_ref, _stream, fields):
        assert fields["release_count"] == 0


def test_no_required_fields_returns_full_fields_dict() -> None:
    """``required_fields=()`` -- the default -- skips per-key validation
    and yields the full ``manifest.fields`` dict."""
    ingest = _ingest()
    source = _from_template_source()
    corpus = _corpus(ingest, source)
    blob = FakeBlob(
        listing=["umls/2026AA/_manifest.json", "umls/2026AA/meta/MRCONSO.RRF"],
        contents={
            "umls/2026AA/_manifest.json": _VALID_MANIFEST.encode("utf-8"),
            "umls/2026AA/meta/MRCONSO.RRF": b"col1,col2\n",
        },
    )

    with read_partition_with_manifest(
        source=source,
        partition_key="2026AA",
        corpus=corpus,
        blob=blob,  # type: ignore[arg-type]
    ) as (_ref, _stream, fields):
        assert set(fields) == {"release_date", "release_version", "release_count"}


# ---------------------------------------------------------------------------
# Manifest prefix derivation (regression test for the rsplit bug)
# ---------------------------------------------------------------------------


def test_multi_level_glob_reads_manifest_from_partition_prefix() -> None:
    """``FromIngestTemplate.source = "subdir/*.csv"`` resolves to a
    blob under a sub-prefix.  The manifest must still be read from the
    partition prefix (``{prefix}/_manifest.json``), NOT the data file's
    parent directory (``{prefix}/subdir/_manifest.json``).

    Regression test for the rsplit approach the helper deliberately
    avoids -- ``fnmatch``'s ``*`` matches across ``/``.
    """
    ingest = _ingest()
    # FromIngestTemplate with a multi-level glob.
    source = _from_template_source(template_source="subdir/*.csv")
    corpus = _corpus(ingest, source)
    blob = FakeBlob(
        listing=[
            "umls/2026AA/_manifest.json",
            "umls/2026AA/subdir/data.csv",
        ],
        contents={
            # Manifest deliberately ONLY at the partition prefix; the
            # helper must not look at {prefix}/subdir/_manifest.json.
            "umls/2026AA/_manifest.json": _VALID_MANIFEST.encode("utf-8"),
            "umls/2026AA/subdir/data.csv": b"col1,col2\n1,2\n",
        },
    )

    with read_partition_with_manifest(
        source=source,
        partition_key="2026AA",
        corpus=corpus,
        blob=blob,  # type: ignore[arg-type]
        required_fields=("release_date",),
    ) as (ref, stream, fields):
        assert ref.path == "umls/2026AA/subdir/data.csv"
        assert fields["release_date"] == "2026-04-26"
        assert stream.read() == b"col1,col2\n1,2\n"


# ---------------------------------------------------------------------------
# Forward-compat: manifest_version > KNOWN_MAX_VERSION
# ---------------------------------------------------------------------------


def test_future_manifest_version_raises_ingest_resolution_error() -> None:
    """Forward-compat check forwarded by ``IngestManifest.read_from``.

    Bypassing the streaming reader (e.g. ``blob.download(...) +
    json.loads(...)``) would silently misinterpret a future-schema
    manifest -- this test pins that the helper does NOT do that.
    """
    ingest = _ingest()
    source = _from_template_source()
    corpus = _corpus(ingest, source)
    future_manifest = _VALID_MANIFEST.replace('"manifest_version": 1', '"manifest_version": 99')
    blob = FakeBlob(
        listing=["umls/2026AA/_manifest.json", "umls/2026AA/meta/MRCONSO.RRF"],
        contents={
            "umls/2026AA/_manifest.json": future_manifest.encode("utf-8"),
            "umls/2026AA/meta/MRCONSO.RRF": b"",
        },
    )

    # The resolver's own manifest-read fires first for the
    # FromIngestTemplate branch; either way, the forward-compat error
    # surfaces as IngestResolutionError before any data stream opens.
    with (
        pytest.raises(IngestResolutionError, match="manifest_version' is 99"),
        read_partition_with_manifest(
            source=source,
            partition_key="2026AA",
            corpus=corpus,
            blob=blob,  # type: ignore[arg-type]
        ),
    ):
        pass


# ---------------------------------------------------------------------------
# Stream close semantics
# ---------------------------------------------------------------------------


def test_stream_closes_on_normal_exit() -> None:
    ingest = _ingest()
    source = _from_template_source()
    corpus = _corpus(ingest, source)
    blob = FakeBlob(
        listing=["umls/2026AA/_manifest.json", "umls/2026AA/meta/MRCONSO.RRF"],
        contents={
            "umls/2026AA/_manifest.json": _VALID_MANIFEST.encode("utf-8"),
            "umls/2026AA/meta/MRCONSO.RRF": b"col1,col2\n",
        },
    )

    with read_partition_with_manifest(
        source=source,
        partition_key="2026AA",
        corpus=corpus,
        blob=blob,  # type: ignore[arg-type]
    ) as (_ref, stream, _fields):
        assert not stream.closed

    # The data stream is the last one opened (manifest opens first).
    data_stream = blob.streams_opened[-1]
    assert data_stream.closed_via_helper


def test_stream_closes_when_with_body_raises() -> None:
    """The context manager must release the blob's HTTP response even
    if the caller's ``with`` body raises -- otherwise a busy pod can
    pressure the connection pool."""
    ingest = _ingest()
    source = _from_template_source()
    corpus = _corpus(ingest, source)
    blob = FakeBlob(
        listing=["umls/2026AA/_manifest.json", "umls/2026AA/meta/MRCONSO.RRF"],
        contents={
            "umls/2026AA/_manifest.json": _VALID_MANIFEST.encode("utf-8"),
            "umls/2026AA/meta/MRCONSO.RRF": b"col1,col2\n",
        },
    )

    class _CallerError(RuntimeError):
        pass

    with (
        pytest.raises(_CallerError),
        read_partition_with_manifest(
            source=source,
            partition_key="2026AA",
            corpus=corpus,
            blob=blob,  # type: ignore[arg-type]
        ),
    ):
        raise _CallerError("boom")

    data_stream = blob.streams_opened[-1]
    assert data_stream.closed_via_helper


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_log_emits_info_with_path_and_sensitivity() -> None:
    """When ``log`` is supplied, the helper emits a single info line
    with the ref path + sensitivity.  Matches the existing pipeline
    convention so authors do not lose their per-asset audit trail."""
    ingest = _ingest()
    source = _from_template_source()
    corpus = _corpus(ingest, source)
    blob = FakeBlob(
        listing=["umls/2026AA/_manifest.json", "umls/2026AA/meta/MRCONSO.RRF"],
        contents={
            "umls/2026AA/_manifest.json": _VALID_MANIFEST.encode("utf-8"),
            "umls/2026AA/meta/MRCONSO.RRF": b"",
        },
    )

    class _RecordingLog:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        def info(self, msg: str, *args: object) -> None:
            self.calls.append((msg, args))

    log = _RecordingLog()

    with read_partition_with_manifest(
        source=source,
        partition_key="2026AA",
        corpus=corpus,
        blob=blob,  # type: ignore[arg-type]
        log=log,
    ):
        pass

    assert len(log.calls) == 1
    msg, args = log.calls[0]
    assert "Reading" in msg
    assert args == ("umls/2026AA/meta/MRCONSO.RRF", "confidential")
