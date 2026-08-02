"""Tests for ``materialize_with_manifest``.

Covers:

- Happy path: pattern returns N results -> dispatcher writes one
  ``_manifest.json`` with N file entries.
- Atomicity: pattern raises -> dispatcher does NOT write the manifest
  (partial-write recovery state).
- Re-run after partial crash: missing files upload, manifest is
  written; an all-skipped second run still rewrites the manifest with
  a fresh ``materialized_at``.
- Manifest content: ``manifest_version: 1``, resolver block, partition
  key, every file entry from the pattern's results.
- Resolver block shape differs by pattern (``api_resolver`` vs
  ``http_urls`` -- audit trail).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from io import BytesIO
from typing import IO, Any
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from moncpipelib.contracts.models import IngestContract
from moncpipelib.ingest.dispatcher import materialize_with_manifest
from moncpipelib.ingest.manifest import IngestManifest
from moncpipelib.ingest.patterns.api_resolver import ApiResolverPattern
from moncpipelib.ingest.types import IngestContext, IngestResult, PartitionSpec


def _parse_manifest(raw: bytes) -> IngestManifest:
    """Helper: parse a manifest blob's bytes via the streaming reader."""
    return IngestManifest.read_from(BytesIO(raw))


# ---------------------------------------------------------------------------
# In-memory blob + stub patterns
# ---------------------------------------------------------------------------


class FakeBlob:
    def __init__(self) -> None:
        self.blobs: dict[str, tuple[bytes, str]] = {}
        self.upload_calls: list[str] = []

    def read_sha256_metadata(self, sensitivity: str, path: str) -> str | None:
        del sensitivity
        entry = self.blobs.get(path)
        return entry[1] if entry else None

    def upload(
        self,
        sensitivity: str,
        path: str,
        data: bytes | IO[bytes],
        sha256: str,
    ) -> None:
        del sensitivity
        self.upload_calls.append(path)
        # Per #239 + #243 the dispatcher streams uploads via a file
        # handle for the manifest path; drain the handle into bytes so
        # the existing test assertions against ``self.blobs[path][0]``
        # keep working.
        body = data if isinstance(data, bytes) else data.read()
        self.blobs[path] = (body, sha256)

    def exists(self, sensitivity: str, path: str) -> bool:
        del sensitivity
        return path in self.blobs

    def download(self, sensitivity: str, path: str) -> bytes:
        del sensitivity
        return self.blobs[path][0]


_UNSET = object()


class _StubPattern:
    """A pattern that emits a fixed result list, no I/O.

    ``partition_metadata`` is added only when the constructor receives a
    non-sentinel ``partition_metadata`` value -- so tests can exercise
    BOTH the back-compat path (pattern lacks the method) and the #256
    path (pattern provides the method) by toggling the kwarg.
    """

    def __init__(
        self,
        name: str,
        results: list[IngestResult] | None = None,
        raise_on_materialize: BaseException | None = None,
        partition_metadata: dict[str, Any] | object = _UNSET,
        manifest_resolver_block: dict[str, Any] | object = _UNSET,
    ) -> None:
        self._name = name
        self._results = results or []
        self._raise = raise_on_materialize
        self.calls = 0
        self.partition_metadata_calls: list[tuple[str]] = []
        if manifest_resolver_block is not _UNSET:
            # Same opt-in binding trick as partition_metadata below --
            # the dispatcher discovers the method via getattr (per #415).
            block: dict[str, Any] = manifest_resolver_block  # type: ignore[assignment]

            def _manifest_resolver_block(contract: IngestContract) -> dict[str, Any]:
                del contract
                return dict(block)

            self.manifest_resolver_block = _manifest_resolver_block
        if partition_metadata is not _UNSET:
            # Bind a method only when requested.  ``getattr(pattern,
            # 'partition_metadata', None)`` in the dispatcher then
            # distinguishes the two cases without forcing the Protocol
            # to grow a new method.
            metadata: dict[str, Any] = partition_metadata  # type: ignore[assignment]

            def _partition_metadata(
                contract: IngestContract,
                partition_key: str,
                ctx: IngestContext,
            ) -> dict[str, Any]:
                del contract, ctx
                self.partition_metadata_calls.append((partition_key,))
                return dict(metadata)

            self.partition_metadata = _partition_metadata

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name

    def discover_partitions(
        self, contract: IngestContract, ctx: IngestContext
    ) -> list[PartitionSpec]:
        del contract, ctx
        return []

    def materialize_partition(
        self,
        contract: IngestContract,
        partition_spec: PartitionSpec,
        blob: FakeBlob,
        ctx: IngestContext,
    ) -> list[IngestResult]:
        del contract, partition_spec, blob, ctx
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return self._results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _http_urls_contract() -> IngestContract:
    return IngestContract(
        source_id="11111111-1111-1111-1111-111111111111",
        source_name="cms-asp",
        sensitivity="public",
        pattern="http_urls",
        prefix_template="cms_asp/{partition_key}",
        extract=("zip",),
        strip_extensions=(),
        pattern_config={"periods": []},
    )


def _api_resolver_contract() -> IngestContract:
    return IngestContract(
        source_id="22222222-2222-2222-2222-222222222222",
        source_name="umls-meta",
        sensitivity="confidential",
        pattern="api_resolver",
        prefix_template="umls/{partition_key}",
        extract=("zip", "zip"),
        strip_extensions=(),
        extract_filter=("meta/**",),
        pattern_config={
            "resolver": "uts_release",
            "resolver_config": {"release_type": "umls-full-release"},
            "credential": {"secret_name": "uts-api-key"},
            "partition": {"mode": "dynamic", "key_from": "release_version"},
        },
        data_owner="data-platform",
        compliance_review="SECURITY.md#umls",
    )


def _spec(key: str = "2026-04-26", **fields: Any) -> PartitionSpec:
    return PartitionSpec(key=key, metadata={"partition_key": key, **fields})


def _ctx() -> IngestContext:
    return IngestContext(log=MagicMock(name="LoggingContext"))


def _result(path: str, *, action: str = "uploaded", size: int = 4) -> IngestResult:
    return IngestResult(path=path, sha256=f"sha-of-{path}", action=action, size_bytes=size)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_writes_manifest_with_all_file_entries() -> None:
    contract = _http_urls_contract()
    spec = _spec("2024-01-01")
    blob = FakeBlob()
    pattern = _StubPattern(
        "http_urls",
        results=[
            _result("cms_asp/2024-01-01/a.csv"),
            _result("cms_asp/2024-01-01/b.csv"),
        ],
    )

    results = materialize_with_manifest(pattern, contract, spec, blob, _ctx())  # type: ignore[arg-type]

    assert len(results) == 2
    manifest_path = "cms_asp/2024-01-01/_manifest.json"
    assert manifest_path in blob.blobs

    parsed = _parse_manifest(blob.blobs[manifest_path][0])
    assert parsed.manifest_version == 1
    assert parsed.source_id == contract.source_id
    assert parsed.source_name == contract.source_name
    assert parsed.partition_key == "2024-01-01"
    assert len(parsed.files) == 2
    paths = {f.path for f in parsed.files}
    assert paths == {"cms_asp/2024-01-01/a.csv", "cms_asp/2024-01-01/b.csv"}


def test_manifest_resolver_block_for_http_urls() -> None:
    """For http_urls, the resolver block carries name=http_urls and
    empty config -- audit symmetry across patterns."""
    contract = _http_urls_contract()
    spec = _spec("2024-01-01")
    blob = FakeBlob()
    pattern = _StubPattern("http_urls", results=[_result("cms_asp/2024-01-01/a.csv")])

    materialize_with_manifest(pattern, contract, spec, blob, _ctx())  # type: ignore[arg-type]

    parsed = _parse_manifest(blob.blobs["cms_asp/2024-01-01/_manifest.json"][0])
    assert parsed.resolver == {"name": "http_urls", "config": {}}


def test_manifest_resolver_block_for_api_resolver() -> None:
    """For api_resolver, the resolver block carries the resolver name
    plus its config (audit trail of which API produced the partition).

    Per #415 the block comes from the pattern's
    ``manifest_resolver_block`` method rather than a dispatcher
    name-check; this test threads the REAL ApiResolverPattern's block
    through a stub (avoiding the real pattern's network I/O) and pins
    the pre-#415 manifest bytes expectation."""
    contract = _api_resolver_contract()
    spec = _spec(
        "2026AA",
        release_version="2026AA",
        download_url="https://upstream/release.zip",
    )
    blob = FakeBlob()
    pattern = _StubPattern(
        "api_resolver",
        results=[_result("umls/2026AA/meta/MRCONSO.RRF")],
        manifest_resolver_block=ApiResolverPattern().manifest_resolver_block(contract),
    )

    materialize_with_manifest(pattern, contract, spec, blob, _ctx())  # type: ignore[arg-type]

    parsed = _parse_manifest(blob.blobs["umls/2026AA/_manifest.json"][0])
    assert parsed.resolver == {
        "name": "uts_release",
        "config": {"release_type": "umls-full-release"},
    }


def test_api_resolver_manifest_resolver_block_method() -> None:
    """Unit: ApiResolverPattern.manifest_resolver_block returns the
    resolver name + resolver_config from the contract, no I/O."""
    block = ApiResolverPattern().manifest_resolver_block(_api_resolver_contract())
    assert block == {
        "name": "uts_release",
        "config": {"release_type": "umls-full-release"},
    }


def test_manifest_resolver_block_from_pattern_method_is_normalized_and_coerced() -> None:
    """A pattern-provided block lands in the manifest with the config
    values JSON-coerced (#233 symmetry) and the name stringified."""
    contract = _http_urls_contract()
    spec = _spec("2024-01-01")
    blob = FakeBlob()
    pattern = _StubPattern(
        "custom_pattern",
        results=[_result("cms_asp/2024-01-01/a.csv")],
        manifest_resolver_block={
            "name": "my_plan",
            "config": {"since": date(2024, 1, 1), "limit": Decimal("5")},
        },
    )

    materialize_with_manifest(pattern, contract, spec, blob, _ctx())  # type: ignore[arg-type]

    parsed = _parse_manifest(blob.blobs["cms_asp/2024-01-01/_manifest.json"][0])
    assert parsed.resolver == {
        "name": "my_plan",
        "config": {"since": "2024-01-01", "limit": "5"},
    }


def test_manifest_resolver_block_fallback_without_pattern_method() -> None:
    """A pattern without manifest_resolver_block gets the generic
    ``{name: <pattern.name>, config: {}}`` block (back-compat)."""
    contract = _http_urls_contract()
    spec = _spec("2024-01-01")
    blob = FakeBlob()
    pattern = _StubPattern("some_pattern", results=[_result("cms_asp/2024-01-01/a.csv")])

    materialize_with_manifest(pattern, contract, spec, blob, _ctx())  # type: ignore[arg-type]

    parsed = _parse_manifest(blob.blobs["cms_asp/2024-01-01/_manifest.json"][0])
    assert parsed.resolver == {"name": "some_pattern", "config": {}}


def test_manifest_fields_carries_partition_metadata() -> None:
    """Back-compat: when the pattern has no ``partition_metadata``
    method, ``manifest.fields`` falls back to ``partition_spec.metadata``.
    This matches dispatcher behavior pre-#256 (and the test stub here
    has no ``partition_metadata``, exercising the fallback path)."""
    contract = _api_resolver_contract()
    spec = _spec(
        "2026AA",
        release_version="2026AA",
        release_date="2026-04-26",
    )
    blob = FakeBlob()
    pattern = _StubPattern("api_resolver", results=[_result("umls/2026AA/x")])

    materialize_with_manifest(pattern, contract, spec, blob, _ctx())  # type: ignore[arg-type]

    parsed = _parse_manifest(blob.blobs["umls/2026AA/_manifest.json"][0])
    assert parsed.fields["release_version"] == "2026AA"
    assert parsed.fields["release_date"] == "2026-04-26"


def test_manifest_fields_sourced_from_partition_metadata_method() -> None:
    """Per #256: when the pattern provides ``partition_metadata``, the
    dispatcher uses its return value (not ``partition_spec.metadata``)
    as the manifest's ``fields`` block.  Models the production path
    where Dagster's dynamic-partitions registry has dropped the
    discovery-time metadata and only the partition_key survives to
    materialize time."""
    contract = _api_resolver_contract()
    # Spec metadata is empty -- this is the production shape, where
    # the asset body builds ``PartitionSpec(key=context.partition_key)``
    # at materialize time and Dagster has not persisted spec metadata.
    spec = PartitionSpec(key="2026AA", metadata={})
    blob = FakeBlob()
    pattern = _StubPattern(
        "api_resolver",
        results=[_result("umls/2026AA/x")],
        partition_metadata={
            "partition_key": "2026AA",
            "release_version": "2026AA",
            "release_date": "2026-04-26",
            "download_url": "https://uts/2026AA.zip",
        },
    )

    materialize_with_manifest(pattern, contract, spec, blob, _ctx())  # type: ignore[arg-type]

    parsed = _parse_manifest(blob.blobs["umls/2026AA/_manifest.json"][0])
    # Regression test for #256: manifest.fields must be populated even
    # though spec.metadata was empty.
    assert parsed.fields == {
        "partition_key": "2026AA",
        "release_version": "2026AA",
        "release_date": "2026-04-26",
        "download_url": "https://uts/2026AA.zip",
    }
    assert pattern.partition_metadata_calls == [("2026AA",)]


def test_manifest_fields_falls_back_to_spec_metadata_when_method_returns_empty() -> None:
    """Defensive: when ``partition_metadata`` returns ``{}`` (e.g. the
    resolver no longer hosts the partition, or a stub returns nothing),
    the dispatcher falls back to ``partition_spec.metadata``.  Lets
    callers that populate spec metadata directly continue to work even
    against a pattern that defines but cannot satisfy
    ``partition_metadata``."""
    contract = _api_resolver_contract()
    spec = _spec("2026AA", release_version="2026AA", release_date="2026-04-26")
    blob = FakeBlob()
    pattern = _StubPattern(
        "api_resolver",
        results=[_result("umls/2026AA/x")],
        partition_metadata={},
    )

    materialize_with_manifest(pattern, contract, spec, blob, _ctx())  # type: ignore[arg-type]

    parsed = _parse_manifest(blob.blobs["umls/2026AA/_manifest.json"][0])
    assert parsed.fields["release_version"] == "2026AA"
    assert parsed.fields["release_date"] == "2026-04-26"


def test_manifest_materialized_at_is_iso8601_utc() -> None:
    contract = _http_urls_contract()
    spec = _spec("2024-01-01")
    blob = FakeBlob()
    pattern = _StubPattern("http_urls", results=[_result("cms_asp/2024-01-01/a.csv")])

    materialize_with_manifest(pattern, contract, spec, blob, _ctx())  # type: ignore[arg-type]

    parsed = _parse_manifest(blob.blobs["cms_asp/2024-01-01/_manifest.json"][0])
    # Format: "YYYY-MM-DDTHH:MM:SSZ"
    assert parsed.materialized_at.endswith("Z")
    assert parsed.materialized_at[10] == "T"


def test_manifest_blob_has_sha256_metadata() -> None:
    """The manifest blob's own bytes are sha256'd before upload so the
    audit trail extends to the manifest itself."""
    contract = _http_urls_contract()
    spec = _spec("2024-01-01")
    blob = FakeBlob()
    pattern = _StubPattern("http_urls", results=[_result("cms_asp/2024-01-01/a.csv")])

    materialize_with_manifest(pattern, contract, spec, blob, _ctx())  # type: ignore[arg-type]

    manifest_path = "cms_asp/2024-01-01/_manifest.json"
    assert blob.read_sha256_metadata("public", manifest_path) is not None


# ---------------------------------------------------------------------------
# Atomicity (partial-write recovery contract per #216)
# ---------------------------------------------------------------------------


def test_pattern_failure_leaves_no_manifest() -> None:
    """If the pattern raises, the dispatcher must NOT write the
    manifest -- the partition is left in the partial-write
    intermediate state.  Re-runs are expected to recover via
    ``hash_compare`` idempotency."""
    contract = _http_urls_contract()
    spec = _spec("2024-01-01")
    blob = FakeBlob()
    pattern = _StubPattern(
        "http_urls", raise_on_materialize=RuntimeError("network failed mid-upload")
    )

    with pytest.raises(RuntimeError, match="network failed"):
        materialize_with_manifest(pattern, contract, spec, blob, _ctx())  # type: ignore[arg-type]

    manifest_path = "cms_asp/2024-01-01/_manifest.json"
    assert manifest_path not in blob.blobs


def test_recovery_after_partial_crash_writes_manifest() -> None:
    """Re-running the dispatcher on a partial partition writes the
    manifest once the pattern returns successfully.  This pins the
    'closing the recovery window' part of the atomicity contract."""
    contract = _http_urls_contract()
    spec = _spec("2024-01-01")
    blob = FakeBlob()

    # First run crashes mid-flight -- no manifest.
    failing = _StubPattern("http_urls", raise_on_materialize=RuntimeError("crash"))
    with pytest.raises(RuntimeError):
        materialize_with_manifest(failing, contract, spec, blob, _ctx())  # type: ignore[arg-type]
    assert "cms_asp/2024-01-01/_manifest.json" not in blob.blobs

    # Second run succeeds -- manifest now lands.
    succeeding = _StubPattern("http_urls", results=[_result("cms_asp/2024-01-01/a.csv")])
    results = materialize_with_manifest(succeeding, contract, spec, blob, _ctx())  # type: ignore[arg-type]
    assert len(results) == 1
    assert "cms_asp/2024-01-01/_manifest.json" in blob.blobs


def test_all_skipped_run_still_writes_manifest_with_fresh_timestamp() -> None:
    """An idempotent re-run (every result == ``"skipped"``) still writes
    the manifest -- the manifest is the canonical 'this partition is
    fully materialized' marker, and re-asserting it on every successful
    run keeps the audit timestamp current."""
    contract = _http_urls_contract()
    spec = _spec("2024-01-01")
    blob = FakeBlob()
    pattern = _StubPattern(
        "http_urls",
        results=[
            _result("cms_asp/2024-01-01/a.csv", action="skipped"),
            _result("cms_asp/2024-01-01/b.csv", action="skipped"),
        ],
    )

    materialize_with_manifest(pattern, contract, spec, blob, _ctx())  # type: ignore[arg-type]

    manifest_path = "cms_asp/2024-01-01/_manifest.json"
    assert manifest_path in blob.blobs


# ---------------------------------------------------------------------------
# Defensive: pattern returning a manifest entry doesn't list itself
# ---------------------------------------------------------------------------


def test_dispatcher_skips_self_reference_in_files_list() -> None:
    """If a pattern ever (incorrectly) emits an IngestResult for the
    manifest itself, the dispatcher filters it out so the manifest
    never lists itself."""
    contract = _http_urls_contract()
    spec = _spec("2024-01-01")
    blob = FakeBlob()
    pattern = _StubPattern(
        "http_urls",
        results=[
            _result("cms_asp/2024-01-01/a.csv"),
            _result("cms_asp/2024-01-01/_manifest.json"),  # self-reference
        ],
    )

    materialize_with_manifest(pattern, contract, spec, blob, _ctx())  # type: ignore[arg-type]

    raw = blob.blobs["cms_asp/2024-01-01/_manifest.json"][0].decode("utf-8")
    data = json.loads(raw)
    paths = {f["path"] for f in data["files"]}
    assert paths == {"cms_asp/2024-01-01/a.csv"}


def test_dispatcher_returns_pattern_results_unchanged() -> None:
    """The dispatcher returns the original IngestResult list unchanged
    so callers that bookkeep against ``action`` / ``size_bytes`` see
    exactly what the pattern produced."""
    contract = _http_urls_contract()
    spec = _spec("2024-01-01")
    blob = FakeBlob()
    expected = [
        _result("cms_asp/2024-01-01/a.csv", action="uploaded"),
        _result("cms_asp/2024-01-01/b.csv", action="skipped"),
    ]
    pattern = _StubPattern("http_urls", results=expected)

    results = materialize_with_manifest(pattern, contract, spec, blob, _ctx())  # type: ignore[arg-type]

    assert results == expected


# ---------------------------------------------------------------------------
# Streaming-memory e2e (#243 / Migration 012 Phase B)
#
# Pins that the dispatcher's manifest write path -- which builds an
# IngestManifest from N IngestResults, hashes it, and uploads the bytes --
# does not buffer the full manifest as a `bytes` object during the
# write+upload sequence.  Peak heap should track the streaming chunk
# size on top of the inevitable dataclass-tuple end state, not double
# it during serialization.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_dispatcher_manifest_write_is_memory_bounded() -> None:
    """50k-result materialize_with_manifest stays bounded.

    Pre-#243 the dispatcher built ``manifest.to_json()`` (a Python str
    holding the full document) THEN ``.encode("utf-8")`` (a bytes copy)
    THEN passed it to ``blob.upload(bytes, ...)``.  Three full copies
    of the serialized manifest in memory simultaneously, plus the
    dataclass tuple -- ~4x the manifest's serialized size at peak.

    Post-#243 the dispatcher streams write_to into a tempfile while
    hashing in the same pass; only the dataclass tuple lives on the
    heap.  This test pins the bound at ~3x the dataclass-tuple size,
    well below the pre-fix peak.
    """
    import tracemalloc

    n_results = 50_000
    threshold_bytes = 32 * 1024 * 1024

    contract = _http_urls_contract()
    spec = _spec("memory-bound")
    blob = FakeBlob()
    pattern = _StubPattern(
        "http_urls",
        results=[_result(f"cms_asp/memory-bound/file_{i:08d}.csv") for i in range(n_results)],
    )

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        materialize_with_manifest(pattern, contract, spec, blob, _ctx())  # type: ignore[arg-type]
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    manifest_path = "cms_asp/memory-bound/_manifest.json"
    assert manifest_path in blob.blobs
    manifest_bytes = blob.blobs[manifest_path][0]
    assert len(manifest_bytes) > 1 * 1024 * 1024  # non-trivial payload

    assert peak <= threshold_bytes, (
        f"peak Python heap was {peak / 1024 / 1024:.1f} MiB during a "
        f"{n_results}-result materialize_with_manifest -- streaming "
        f"regression?  Threshold: {threshold_bytes / 1024 / 1024:.0f} MiB."
    )


# ---------------------------------------------------------------------------
# Resolver-config coercion (issue #233)
#
# PyYAML parses bare ISO dates / timestamps in YAML contracts as ``date``
# / ``datetime`` objects, which are not JSON-serializable.  The dispatcher
# coerces ``resolver_config`` to JSON-friendly scalars before constructing
# the manifest so (a) the manifest write does not crash, and (b) the
# in-memory manifest equals what comes back from ``read_from``.
# ---------------------------------------------------------------------------


def _api_resolver_contract_with_resolver_config(
    resolver_config: dict[str, Any],
) -> IngestContract:
    return IngestContract(
        source_id="33333333-3333-3333-3333-333333333333",
        source_name="fda-ndc",
        sensitivity="public",
        pattern="api_resolver",
        prefix_template="fda_ndc/{partition_key}",
        extract=("zip",),
        strip_extensions=(),
        pattern_config={
            "resolver": "calendar",
            "resolver_config": resolver_config,
            "partition": {"mode": "dynamic", "key_from": "snapshot_date"},
        },
        data_owner="data-platform",
        compliance_review="SECURITY.md#fda-ndc",
    )


def test_manifest_coerces_date_in_resolver_config() -> None:
    """Reproduces issue #233: a ``date`` from PyYAML in resolver_config
    must serialize to an ISO string, not crash ``json.dumps``."""
    contract = _api_resolver_contract_with_resolver_config(
        {
            "start_date": date(2024, 1, 1),
            "cadence": "weekly",
            "anchor_dow": "Sunday",
            "url": "https://download.open.fda.gov/drug/ndc/...",
        }
    )
    spec = _spec("2024-01-07")
    blob = FakeBlob()
    pattern = _StubPattern(
        "api_resolver",
        results=[_result("fda_ndc/2024-01-07/ndc.zip")],
        manifest_resolver_block=ApiResolverPattern().manifest_resolver_block(contract),
    )

    materialize_with_manifest(pattern, contract, spec, blob, _ctx())  # type: ignore[arg-type]

    parsed = _parse_manifest(blob.blobs["fda_ndc/2024-01-07/_manifest.json"][0])
    assert parsed.resolver["config"]["start_date"] == "2024-01-01"
    # Other scalar fields untouched.
    assert parsed.resolver["config"]["cadence"] == "weekly"


def test_manifest_coerces_nested_date_in_resolver_config() -> None:
    """Coercion must recurse: a ``date`` inside a nested dict or list
    in resolver_config is just as likely to come from PyYAML."""
    contract = _api_resolver_contract_with_resolver_config(
        {
            "windows": [
                {"from": date(2024, 1, 1), "to": date(2024, 6, 30)},
                {"from": date(2024, 7, 1), "to": date(2024, 12, 31)},
            ],
        }
    )
    spec = _spec("2024-01-07")
    blob = FakeBlob()
    pattern = _StubPattern(
        "api_resolver",
        results=[_result("fda_ndc/2024-01-07/ndc.zip")],
        manifest_resolver_block=ApiResolverPattern().manifest_resolver_block(contract),
    )

    materialize_with_manifest(pattern, contract, spec, blob, _ctx())  # type: ignore[arg-type]

    parsed = _parse_manifest(blob.blobs["fda_ndc/2024-01-07/_manifest.json"][0])
    windows = parsed.resolver["config"]["windows"]
    assert windows == [
        {"from": "2024-01-01", "to": "2024-06-30"},
        {"from": "2024-07-01", "to": "2024-12-31"},
    ]


def test_manifest_coerces_datetime_uuid_decimal_in_resolver_config() -> None:
    """Same coercion applies to ``datetime``, ``UUID``, ``Decimal`` --
    realistic shapes in resolver_config across calendars and APIs."""
    contract = _api_resolver_contract_with_resolver_config(
        {
            "cutover_at": datetime(2024, 1, 1, 12, 30, tzinfo=UTC),
            "upstream_id": UUID("12345678-1234-5678-1234-567812345678"),
            "threshold": Decimal("0.95"),
        }
    )
    spec = _spec("2024-01-07")
    blob = FakeBlob()
    pattern = _StubPattern(
        "api_resolver",
        results=[_result("fda_ndc/2024-01-07/ndc.zip")],
        manifest_resolver_block=ApiResolverPattern().manifest_resolver_block(contract),
    )

    materialize_with_manifest(pattern, contract, spec, blob, _ctx())  # type: ignore[arg-type]

    parsed = _parse_manifest(blob.blobs["fda_ndc/2024-01-07/_manifest.json"][0])
    cfg = parsed.resolver["config"]
    assert cfg["cutover_at"] == "2024-01-01T12:30:00+00:00"
    assert cfg["upstream_id"] == "12345678-1234-5678-1234-567812345678"
    assert cfg["threshold"] == "0.95"


# ---------------------------------------------------------------------------
# partition_spec.metadata coercion (symmetric with resolver_config; #248
# review).  PyYAML / Dagster partition definitions can place dates,
# datetimes, UUIDs, or Decimals in ``partition_spec.metadata``.  The
# in-memory dataclass must round-trip equal to what's read back from
# the manifest, so the coercion must apply on BOTH sides of
# `_build_manifest`.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metadata_field", "raw_value", "expected_serialized"),
    [
        pytest.param("snapshot_date", date(2024, 1, 1), "2024-01-01", id="date"),
        pytest.param(
            "cutover_at",
            datetime(2024, 1, 1, 12, 30, tzinfo=UTC),
            "2024-01-01T12:30:00+00:00",
            id="datetime",
        ),
        pytest.param(
            "upstream_id",
            UUID("12345678-1234-5678-1234-567812345678"),
            "12345678-1234-5678-1234-567812345678",
            id="uuid",
        ),
        pytest.param("threshold", Decimal("0.95"), "0.95", id="decimal"),
    ],
)
def test_manifest_coerces_partition_metadata(
    metadata_field: str, raw_value: Any, expected_serialized: Any
) -> None:
    """``_build_manifest`` runs ``_coerce_jsonable`` on
    ``partition_spec.metadata`` symmetrically with ``resolver_config``.

    Without this coercion the on-disk write would still succeed
    (rescued by ``_json_default``), but the in-memory dataclass would
    not round-trip equal to the parsed manifest -- the same #233-class
    authoring trap, just on a different field.

    Pinning all four rescued types here so any future schema-additive
    change to coercion behavior surfaces a regression at PR time.
    """
    contract = _api_resolver_contract_with_resolver_config({})
    # Build a spec whose metadata carries the rescued type.
    spec = PartitionSpec(
        key="2024-01-07",
        metadata={"partition_key": "2024-01-07", metadata_field: raw_value},
    )
    blob = FakeBlob()
    pattern = _StubPattern("api_resolver", results=[_result("fda_ndc/2024-01-07/ndc.zip")])

    materialize_with_manifest(pattern, contract, spec, blob, _ctx())  # type: ignore[arg-type]

    parsed = _parse_manifest(blob.blobs["fda_ndc/2024-01-07/_manifest.json"][0])
    assert parsed.fields[metadata_field] == expected_serialized
    # And the round-trip-equality property: the parsed manifest's
    # fields dict equals the same dict we'd build from the spec
    # metadata after coercion (no rescued types lurking).
    assert parsed.fields == {
        "partition_key": "2024-01-07",
        metadata_field: expected_serialized,
    }


def test_manifest_coerces_nested_partition_metadata() -> None:
    """Coercion recurses through nested dicts / lists in partition
    metadata too -- a pattern might place a list of dates or a
    daterange dict and we should not crash or drop the round-trip
    invariant."""
    contract = _api_resolver_contract_with_resolver_config({})
    spec = PartitionSpec(
        key="2024-q1",
        metadata={
            "partition_key": "2024-q1",
            "windows": [
                {"from": date(2024, 1, 1), "to": date(2024, 3, 31)},
                {"from": date(2024, 4, 1), "to": date(2024, 6, 30)},
            ],
        },
    )
    blob = FakeBlob()
    pattern = _StubPattern("api_resolver", results=[_result("fda_ndc/2024-q1/x.zip")])

    materialize_with_manifest(pattern, contract, spec, blob, _ctx())  # type: ignore[arg-type]

    parsed = _parse_manifest(blob.blobs["fda_ndc/2024-q1/_manifest.json"][0])
    assert parsed.fields["windows"] == [
        {"from": "2024-01-01", "to": "2024-03-31"},
        {"from": "2024-04-01", "to": "2024-06-30"},
    ]
