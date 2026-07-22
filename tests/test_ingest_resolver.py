"""Tests for ``resolve_source_for_partition``.

Phase 1 cases (legacy / static / drift detection) plus Phase 2 cases
covering the ``FromIngestTemplate`` branch (manifest-driven hydration,
manifest-version-too-new rejection, missing-field detection,
manifest-absent partial-write detection, ``{field}`` substitution in
the template's source glob).
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
    BlobRef,
    IngestResolutionError,
    RawUrl,
    resolve_source_for_partition,
)


class FakeBlob:
    """In-memory blob stand-in.

    Implements the methods the resolver actually calls in either
    branch: ``list``, ``exists``, ``stream`` (used by the streaming
    manifest read path), ``download`` (legacy small-payload reads).
    """

    def __init__(
        self,
        listing: list[str] | None = None,
        contents: dict[str, bytes] | None = None,
    ) -> None:
        self._listing = list(listing or [])
        self._contents = dict(contents or {})

    def list(self, sensitivity: str, prefix: str) -> list[str]:
        del sensitivity, prefix
        return list(self._listing)

    def iter_list(self, sensitivity: str, prefix: str) -> Iterator[str]:
        """Forward-only iterator over the listing (#246)."""
        del sensitivity, prefix
        return iter(self._listing)

    def exists(self, sensitivity: str, path: str) -> bool:
        del sensitivity
        return path in self._contents

    def download(self, sensitivity: str, path: str) -> bytes:
        del sensitivity
        return self._contents[path]

    def stream(self, sensitivity: str, path: str) -> IO[bytes]:
        """Forward-only file-like over the blob's contents (#241)."""
        del sensitivity
        return BytesIO(self._contents[path])


def _ingest(
    source_name: str = "cms-asp",
    prefix_template: str = "cms_asp/{partition_key}",
    sensitivity: Literal["public", "confidential", "phi"] = "public",
    pattern: str = "http_urls",
    pattern_config: dict[str, object] | None = None,
) -> IngestContract:
    return IngestContract(
        source_id="11111111-1111-1111-1111-111111111111",
        source_name=source_name,
        sensitivity=sensitivity,
        pattern=pattern,
        prefix_template=prefix_template,
        extract=("zip",),
        strip_extensions=(),
        pattern_config=pattern_config
        or {"periods": [{"partition_key": "2024-01-01", "urls": ["https://example.com/x.zip"]}]},
    )


def _source_static(ingest_source: str | None = "cms-asp") -> DataSource:
    return DataSource(
        source_id="22222222-2222-2222-2222-222222222222",
        source_name="cms-asp-crosswalk",
        periods=[
            Period(
                source="*crosswalk*.csv",
                effective_from=date(2024, 1, 1),
                partition_key="2024-01-01",
            )
        ],
        ingest_source=ingest_source,
    )


def _corpus(ingest: IngestContract, source: DataSource) -> ContractCorpus:
    return ContractCorpus(
        ingests={ingest.source_name: ingest},
        sources={source.source_name: source},
    )


# ---------------------------------------------------------------------------
# Phase 1 regression cases
# ---------------------------------------------------------------------------


def test_legacy_source_returns_raw_url() -> None:
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

    result = resolve_source_for_partition(source, "2024-01-01", corpus, FakeBlob())  # type: ignore[arg-type]

    assert result == [RawUrl("https://legacy.example.com/file.csv")]


def test_static_single_match_returns_blob_ref() -> None:
    ingest = _ingest()
    source = _source_static()
    corpus = _corpus(ingest, source)
    blob = FakeBlob(listing=["cms_asp/2024-01-01/crosswalk_q1.csv"])

    result = resolve_source_for_partition(source, "2024-01-01", corpus, blob)  # type: ignore[arg-type]

    assert result == [BlobRef(sensitivity="public", path="cms_asp/2024-01-01/crosswalk_q1.csv")]


def test_static_zero_match_raises() -> None:
    ingest = _ingest()
    source = _source_static()
    corpus = _corpus(ingest, source)
    blob = FakeBlob()

    with pytest.raises(IngestResolutionError, match="got 0"):
        resolve_source_for_partition(source, "2024-01-01", corpus, blob)  # type: ignore[arg-type]


def test_static_multi_match_raises() -> None:
    ingest = _ingest()
    source = _source_static()
    corpus = _corpus(ingest, source)
    blob = FakeBlob(
        listing=[
            "cms_asp/2024-01-01/crosswalk_a.csv",
            "cms_asp/2024-01-01/crosswalk_b.csv",
        ]
    )

    with pytest.raises(IngestResolutionError, match="got 2"):
        resolve_source_for_partition(source, "2024-01-01", corpus, blob)  # type: ignore[arg-type]


def test_static_match_excludes_manifest_blob() -> None:
    """A permissive glob must not collide with the partition's
    ``_manifest.json`` -- the resolver explicitly excludes it from
    candidates so a glob like ``*`` still resolves to the data file."""
    ingest = _ingest()
    source = DataSource(
        source_id="55555555-5555-5555-5555-555555555555",
        source_name="cms-asp-permissive",
        periods=[
            Period(
                source="*",  # matches everything
                effective_from=date(2024, 1, 1),
                partition_key="2024-01-01",
            )
        ],
        ingest_source="cms-asp",
    )
    corpus = _corpus(ingest, source)
    blob = FakeBlob(
        listing=[
            "cms_asp/2024-01-01/_manifest.json",
            "cms_asp/2024-01-01/data.csv",
        ]
    )

    [ref] = resolve_source_for_partition(source, "2024-01-01", corpus, blob)  # type: ignore[arg-type]
    assert isinstance(ref, BlobRef)
    assert ref.path == "cms_asp/2024-01-01/data.csv"


def test_missing_partition_on_source_raises() -> None:
    ingest = _ingest()
    source = _source_static()
    corpus = _corpus(ingest, source)
    blob = FakeBlob(listing=["cms_asp/2024-04-01/crosswalk.csv"])

    with pytest.raises(IngestResolutionError, match="no period"):
        resolve_source_for_partition(source, "2024-04-01", corpus, blob)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FromIngestTemplate branch (Phase 2 manifest reader)
# ---------------------------------------------------------------------------


def _api_resolver_ingest() -> IngestContract:
    return _ingest(
        source_name="umls-meta",
        prefix_template="umls/{partition_key}",
        sensitivity="confidential",
        pattern="api_resolver",
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
) -> DataSource:
    return DataSource(
        source_id="44444444-4444-4444-4444-444444444444",
        source_name="rxnorm-mrconso",
        periods=FromIngestTemplate(
            source=template_source,
            effective_from_field=effective_from_field,
        ),
        ingest_source="umls-meta",
    )


_VALID_MANIFEST = """{
  "fields": {
    "release_date": "2026-04-26",
    "release_version": "2026AA"
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


def test_from_template_happy_path_returns_blob_ref() -> None:
    ingest = _api_resolver_ingest()
    source = _from_template_source()
    corpus = _corpus(ingest, source)
    blob = FakeBlob(
        listing=[
            "umls/2026AA/_manifest.json",
            "umls/2026AA/meta/MRCONSO.RRF",
        ],
        contents={"umls/2026AA/_manifest.json": _VALID_MANIFEST.encode("utf-8")},
    )

    [ref] = resolve_source_for_partition(source, "2026AA", corpus, blob)  # type: ignore[arg-type]

    assert isinstance(ref, BlobRef)
    assert ref.sensitivity == "confidential"
    assert ref.path == "umls/2026AA/meta/MRCONSO.RRF"


def test_from_template_substitutes_manifest_fields_into_glob() -> None:
    """``FromIngestTemplate.source`` may reference manifest fields via
    ``{field_name}`` placeholders; the resolver hydrates them at
    resolve time so a single template can target per-release file
    paths."""
    ingest = _api_resolver_ingest()
    source = _from_template_source(
        template_source="data/{release_version}/MRCONSO.RRF",
    )
    corpus = _corpus(ingest, source)
    blob = FakeBlob(
        listing=[
            "umls/2026AA/_manifest.json",
            "umls/2026AA/data/2026AA/MRCONSO.RRF",
        ],
        contents={"umls/2026AA/_manifest.json": _VALID_MANIFEST.encode("utf-8")},
    )

    [ref] = resolve_source_for_partition(source, "2026AA", corpus, blob)  # type: ignore[arg-type]
    assert ref.path == "umls/2026AA/data/2026AA/MRCONSO.RRF"


def test_from_template_missing_manifest_raises_partial_write() -> None:
    """A partition with files but no manifest is the
    intermediate state for the partial-write atomicity contract.
    The resolver MUST raise so consumers are NOT silently served
    a partial partition."""
    ingest = _api_resolver_ingest()
    source = _from_template_source()
    corpus = _corpus(ingest, source)
    blob = FakeBlob(
        listing=["umls/2026AA/meta/MRCONSO.RRF"],
        contents={},  # no manifest
    )

    with pytest.raises(IngestResolutionError, match="partial-write"):
        resolve_source_for_partition(source, "2026AA", corpus, blob)  # type: ignore[arg-type]


def test_from_template_future_manifest_version_raises() -> None:
    """A manifest with ``manifest_version > KNOWN_MAX_VERSION`` must
    fail loud rather than silently misinterpret the schema.
    Forwarded by ``IngestManifest.read_from``."""
    ingest = _api_resolver_ingest()
    source = _from_template_source()
    corpus = _corpus(ingest, source)
    future_manifest = _VALID_MANIFEST.replace('"manifest_version": 1', '"manifest_version": 99')
    blob = FakeBlob(
        listing=["umls/2026AA/_manifest.json"],
        contents={"umls/2026AA/_manifest.json": future_manifest.encode("utf-8")},
    )

    with pytest.raises(IngestResolutionError, match="manifest_version' is 99"):
        resolve_source_for_partition(source, "2026AA", corpus, blob)  # type: ignore[arg-type]


def test_from_template_missing_effective_from_field_raises() -> None:
    """A manifest that does not carry the field a downstream consumer
    references via ``effective_from_field`` is drift -- a renamed
    resolver field that breaks the consumer.  Resolver must surface
    this rather than silently returning ``None``."""
    ingest = _api_resolver_ingest()
    source = _from_template_source(effective_from_field="release_iso")
    corpus = _corpus(ingest, source)
    blob = FakeBlob(
        listing=["umls/2026AA/meta/MRCONSO.RRF"],
        contents={"umls/2026AA/_manifest.json": _VALID_MANIFEST.encode("utf-8")},
    )

    with pytest.raises(IngestResolutionError, match="release_iso"):
        resolve_source_for_partition(source, "2026AA", corpus, blob)  # type: ignore[arg-type]


def test_from_template_unknown_substitution_field_raises() -> None:
    """A glob that references a manifest field which does not exist in
    ``manifest.fields`` must fail with a clear error -- contract author
    typo at deploy / maintenance time, not silent zero-match."""
    ingest = _api_resolver_ingest()
    source = _from_template_source(
        template_source="data/{not_a_field}/MRCONSO.RRF",
    )
    corpus = _corpus(ingest, source)
    blob = FakeBlob(
        listing=["umls/2026AA/_manifest.json"],
        contents={"umls/2026AA/_manifest.json": _VALID_MANIFEST.encode("utf-8")},
    )

    with pytest.raises(IngestResolutionError, match="not_a_field"):
        resolve_source_for_partition(source, "2026AA", corpus, blob)  # type: ignore[arg-type]


def test_from_template_zero_match_after_substitution_raises() -> None:
    """The exactly-one-match contract holds for the FromIngestTemplate
    branch too (drift detection on the consumer side)."""
    ingest = _api_resolver_ingest()
    source = _from_template_source(
        template_source="meta/MISSING.RRF",  # not in listing
    )
    corpus = _corpus(ingest, source)
    blob = FakeBlob(
        listing=["umls/2026AA/_manifest.json", "umls/2026AA/meta/MRCONSO.RRF"],
        contents={"umls/2026AA/_manifest.json": _VALID_MANIFEST.encode("utf-8")},
    )

    with pytest.raises(IngestResolutionError, match="got 0"):
        resolve_source_for_partition(source, "2026AA", corpus, blob)  # type: ignore[arg-type]
