"""Tests for BlobMirrorPattern (#437).

The foreign source and the destination blob are in-memory fakes -- no
Azure SDK, no network.  We exercise the pattern's real logic: object
matching + meta-file exclusion, the etag-compare idempotency (skip with
no re-download), re-upload on etag change, folder-derived discovery,
duplicate-basename / zero-match guards, the manifest write through the
dispatcher, and the SP credential factory.
"""

from __future__ import annotations

import logging
from io import BytesIO
from types import SimpleNamespace
from typing import IO, Literal
from unittest.mock import MagicMock, patch

import pytest

from moncpipelib.contracts.models import IngestContract
from moncpipelib.ingest.dispatcher import materialize_with_manifest
from moncpipelib.ingest.exceptions import IngestResolutionError
from moncpipelib.ingest.patterns.blob_mirror import (
    BlobMirrorPattern,
    open_foreign_source,
)
from moncpipelib.ingest.types import IngestContext, PartitionSpec


class FakeForeignSource:
    """In-memory ForeignBlobReader stand-in."""

    def __init__(
        self,
        objects: dict[str, tuple[bytes, str]] | None = None,
        child_prefixes: list[str] | None = None,
    ) -> None:
        # path -> (content, etag)
        self.objects = dict(objects or {})
        self._child_prefixes = list(child_prefixes or [])
        self.stream_calls: list[str] = []

    def iter_list(self, prefix: str):
        for path in sorted(self.objects):
            if path.startswith(prefix):
                yield path

    def iter_child_prefixes(self, prefix: str):
        for child in self._child_prefixes:
            if child.startswith(prefix):
                yield child

    def stream(self, path: str) -> IO[bytes]:
        self.stream_calls.append(path)
        return BytesIO(self.objects[path][0])

    def get_properties(self, path: str):
        content, etag = self.objects[path]
        return SimpleNamespace(etag=etag, size=len(content))


class FakeBlob:
    """In-memory BlobStorageResource stand-in with metadata support."""

    def __init__(self) -> None:
        # path -> (content, metadata)
        self.blobs: dict[str, tuple[bytes, dict[str, str]]] = {}

    def upload(
        self,
        sensitivity: str,
        path: str,
        data: bytes | IO[bytes],
        sha256: str,
        extra_metadata: dict[str, str] | None = None,
    ) -> None:
        del sensitivity
        body = data if isinstance(data, bytes) else data.read()
        metadata = {**(extra_metadata or {}), "sha256": sha256}
        self.blobs[path] = (body, metadata)

    def read_sha256_metadata(self, sensitivity: str, path: str) -> str | None:
        return self.read_metadata_value(sensitivity, path, "sha256")

    def read_metadata_value(self, sensitivity: str, path: str, key: str) -> str | None:
        del sensitivity
        entry = self.blobs.get(path)
        if entry is None:
            return None
        target = key.lower()
        for k, v in entry[1].items():
            if k.lower() == target:
                return v
        return None


def _contract(
    *,
    sensitivity: Literal["public", "confidential", "phi"] = "confidential",
    object_glob: str = "*.parquet",
    exclude_globs: list[str] | None = None,
    credential: dict[str, str] | None = None,
    discovery_prefix: str | None = None,
    partition_pattern: str | None = None,
) -> IngestContract:
    source: dict[str, object] = {
        "account_url": "https://examplestorageacct.blob.core.windows.net",
        "container": "delivery",
        "object_prefix": "{partition_key}/visits_oncology",
    }
    if discovery_prefix is not None:
        source["discovery_prefix"] = discovery_prefix
    if partition_pattern is not None:
        source["partition_pattern"] = partition_pattern
    pattern_config: dict[str, object] = {"source": source, "object_glob": object_glob}
    if exclude_globs is not None:
        pattern_config["exclude_globs"] = exclude_globs
    if credential is not None:
        pattern_config["credential"] = credential
    return IngestContract(
        source_id="11111111-1111-1111-1111-111111111111",
        source_name="trilliant-visits-oncology",
        sensitivity=sensitivity,
        pattern="blob_mirror",
        prefix_template="trilliant/visits_oncology/{partition_key}",
        extract=(),
        strip_extensions=(),
        pattern_config=pattern_config,
        data_owner="vp-data-platform",
        compliance_review="SECURITY.md#trilliant",
    )


def _ctx() -> IngestContext:
    return IngestContext(log=logging.getLogger("moncpipelib.test.blob_mirror"))


def _pattern(source: FakeForeignSource) -> BlobMirrorPattern:
    return BlobMirrorPattern(source_factory=lambda *_: source)


def _spec(key: str = "202501") -> PartitionSpec:
    return PartitionSpec(key=key, metadata={"partition_key": key})


# ---------------------------------------------------------------------------
# materialize
# ---------------------------------------------------------------------------


def test_mirrors_all_parts_and_excludes_meta_files() -> None:
    source = FakeForeignSource(
        objects={
            "202501/visits_oncology/part-00001.snappy.parquet": (b"aaa", '"e1"'),
            "202501/visits_oncology/part-00002.snappy.parquet": (b"bbbb", '"e2"'),
            "202501/visits_oncology/_SUCCESS": (b"", '"e3"'),
            "202501/visits_oncology/_committed_123": (b"x", '"e4"'),
        }
    )
    blob = FakeBlob()
    results = _pattern(source).materialize_partition(
        _contract(exclude_globs=["_*"]),
        _spec(),
        blob,
        _ctx(),  # type: ignore[arg-type]
    )

    paths = sorted(r.path for r in results)
    assert paths == [
        "trilliant/visits_oncology/202501/part-00001.snappy.parquet",
        "trilliant/visits_oncology/202501/part-00002.snappy.parquet",
    ]
    assert all(r.action == "uploaded" for r in results)
    # meta-files never landed
    assert "trilliant/visits_oncology/202501/_SUCCESS" not in blob.blobs
    # source etag stamped on our blob metadata
    body, meta = blob.blobs["trilliant/visits_oncology/202501/part-00001.snappy.parquet"]
    assert body == b"aaa"
    assert meta["source_etag"] == "e1"  # quotes stripped
    assert "sha256" in meta


def test_glob_only_matches_parquet() -> None:
    source = FakeForeignSource(
        objects={
            "202501/visits_oncology/part-00001.snappy.parquet": (b"aaa", '"e1"'),
            "202501/visits_oncology/notes.txt": (b"hello", '"e5"'),
        }
    )
    blob = FakeBlob()
    results = _pattern(source).materialize_partition(
        _contract(),
        _spec(),
        blob,
        _ctx(),  # type: ignore[arg-type]
    )
    assert [r.path for r in results] == [
        "trilliant/visits_oncology/202501/part-00001.snappy.parquet"
    ]


def test_rerun_skips_via_etag_without_downloading() -> None:
    source = FakeForeignSource(
        objects={
            "202501/visits_oncology/part-00001.snappy.parquet": (b"aaa", '"e1"'),
            "202501/visits_oncology/part-00002.snappy.parquet": (b"bbbb", '"e2"'),
        }
    )
    blob = FakeBlob()
    pattern = _pattern(source)
    pattern.materialize_partition(_contract(), _spec(), blob, _ctx())  # type: ignore[arg-type]
    assert len(source.stream_calls) == 2  # both downloaded on first run

    source.stream_calls.clear()
    results = pattern.materialize_partition(_contract(), _spec(), blob, _ctx())  # type: ignore[arg-type]

    assert all(r.action == "skipped" for r in results)
    assert source.stream_calls == []  # etag matched -> no re-download
    # sha256 preserved on the skipped result (read from stored metadata)
    assert all(r.sha256 for r in results)


def test_changed_etag_triggers_reupload() -> None:
    contract = _contract()
    blob = FakeBlob()
    source1 = FakeForeignSource(
        objects={"202501/visits_oncology/part-00001.snappy.parquet": (b"old", '"e1"')}
    )
    BlobMirrorPattern(source_factory=lambda *_: source1).materialize_partition(
        contract,
        _spec(),
        blob,
        _ctx(),  # type: ignore[arg-type]
    )

    # New snapshot: same path, new content + etag.
    source2 = FakeForeignSource(
        objects={"202501/visits_oncology/part-00001.snappy.parquet": (b"new-bytes", '"e9"')}
    )
    results = BlobMirrorPattern(source_factory=lambda *_: source2).materialize_partition(
        contract, _spec(), blob, _ctx()
    )  # type: ignore[arg-type]

    assert [r.action for r in results] == ["uploaded"]
    body, meta = blob.blobs["trilliant/visits_oncology/202501/part-00001.snappy.parquet"]
    assert body == b"new-bytes"
    assert meta["source_etag"] == "e9"
    assert source2.stream_calls == ["202501/visits_oncology/part-00001.snappy.parquet"]


def test_zero_matches_raises() -> None:
    source = FakeForeignSource(objects={"202501/visits_oncology/_SUCCESS": (b"", '"e1"')})
    with pytest.raises(IngestResolutionError, match="no objects matching"):
        _pattern(source).materialize_partition(
            _contract(exclude_globs=["_*"]),
            _spec(),
            FakeBlob(),
            _ctx(),  # type: ignore[arg-type]
        )


def test_duplicate_basename_raises() -> None:
    source = FakeForeignSource(
        objects={
            "202501/visits_oncology/a/part.parquet": (b"a", '"e1"'),
            "202501/visits_oncology/b/part.parquet": (b"b", '"e2"'),
        }
    )
    with pytest.raises(IngestResolutionError, match="same landing filename"):
        _pattern(source).materialize_partition(
            _contract(),
            _spec(),
            FakeBlob(),
            _ctx(),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def test_discover_partitions_folder_derived() -> None:
    source = FakeForeignSource(child_prefixes=["202501/", "202502/", "current.json/", "garbage/"])
    contract = _contract(discovery_prefix="", partition_pattern=r"^\d{6}$")
    specs = _pattern(source).discover_partitions(contract, _ctx())
    assert [s.key for s in specs] == ["202501", "202502"]
    assert all(s.metadata == {"partition_key": s.key} for s in specs)


def test_discover_partitions_empty_without_discovery_prefix() -> None:
    source = FakeForeignSource(child_prefixes=["202501/"])
    specs = _pattern(source).discover_partitions(_contract(), _ctx())
    assert specs == []


# ---------------------------------------------------------------------------
# manifest write through the dispatcher
# ---------------------------------------------------------------------------


def test_manifest_written_through_dispatcher() -> None:
    source = FakeForeignSource(
        objects={
            "202501/visits_oncology/part-00001.snappy.parquet": (b"aaa", '"e1"'),
            "202501/visits_oncology/part-00002.snappy.parquet": (b"bbbb", '"e2"'),
        }
    )
    blob = FakeBlob()
    results = materialize_with_manifest(
        _pattern(source),
        _contract(),
        _spec(),
        blob,
        _ctx(),  # type: ignore[arg-type]
    )
    assert len(results) == 2
    manifest_path = "trilliant/visits_oncology/202501/_manifest.json"
    assert manifest_path in blob.blobs
    import json

    manifest = json.loads(blob.blobs[manifest_path][0])
    assert manifest["source_name"] == "trilliant-visits-oncology"
    assert manifest["partition_key"] == "202501"
    assert manifest["fields"] == {"partition_key": "202501"}
    assert manifest["resolver"]["name"] == "blob_mirror"
    assert {f["path"] for f in manifest["files"]} == {
        "trilliant/visits_oncology/202501/part-00001.snappy.parquet",
        "trilliant/visits_oncology/202501/part-00002.snappy.parquet",
    }


# ---------------------------------------------------------------------------
# credential factory
# ---------------------------------------------------------------------------


def test_open_foreign_source_builds_client_secret_credential() -> None:
    secrets = MagicMock()
    secrets.get_secret.return_value = "s3cr3t"
    ctx = IngestContext(log=logging.getLogger("moncpipelib.test.blob_mirror"), secrets=secrets)
    source_cfg = {
        "account_url": "https://examplestorageacct.blob.core.windows.net",
        "container": "delivery",
    }
    credential_cfg = {
        "secret_name": "trilliant-sp",
        "tenant_id": "partner-tenant",
        "client_id": "our-sp",
    }
    with patch(
        "moncpipelib.ingest.patterns.blob_mirror.ForeignBlobSource.from_client_secret"
    ) as mock_factory:
        open_foreign_source(source_cfg, credential_cfg, ctx)

    secrets.get_secret.assert_called_once_with("trilliant-sp")
    mock_factory.assert_called_once_with(
        account_url="https://examplestorageacct.blob.core.windows.net",
        container="delivery",
        tenant_id="partner-tenant",
        client_id="our-sp",
        client_secret="s3cr3t",
    )


def test_open_foreign_source_requires_secrets_resource() -> None:
    ctx = IngestContext(log=logging.getLogger("moncpipelib.test.blob_mirror"))
    with pytest.raises(IngestResolutionError, match="no secrets resource"):
        open_foreign_source(
            {"account_url": "https://x", "container": "c"},
            {"secret_name": "s", "tenant_id": "t", "client_id": "c"},
            ctx,
        )


def test_open_foreign_source_falls_back_to_default_credential() -> None:
    ctx = IngestContext(log=logging.getLogger("moncpipelib.test.blob_mirror"))
    with patch(
        "moncpipelib.ingest.patterns.blob_mirror.ForeignBlobSource.with_default_credential"
    ) as mock_factory:
        open_foreign_source({"account_url": "https://x", "container": "c"}, {}, ctx)
    mock_factory.assert_called_once_with(account_url="https://x", container="c")
