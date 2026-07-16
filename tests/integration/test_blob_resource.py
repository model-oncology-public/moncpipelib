"""Integration tests for BlobStorageResource against Azurite.

These tests exercise the full read/write round-trip against a real
(emulated) Azure Storage surface. The unit tests in
``tests/test_blob_resource.py`` cover the case-insensitive metadata
contract directly; these tests catch regressions in container
resolution, SDK upgrade behavior, and overall wiring.

Regression context: see issue #214. ``read_sha256_metadata`` was
silently broken because Azure Storage normalized the
``x-ms-meta-sha256`` header to ``Sha256`` on read while the lookup
was case-sensitive. The unit tests guard the lookup; this file
guards the round-trip.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from moncpipelib.resources.blob import BlobStorageResource

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration


def test_upload_then_read_sha256_round_trip(blob_resource: BlobStorageResource) -> None:
    """Upload a blob and verify ``read_sha256_metadata`` returns the hash."""
    payload = b"hello-world-payload"
    sha = hashlib.sha256(payload).hexdigest()

    blob_resource.upload("public", "round-trip/foo.bin", payload, sha256=sha)

    assert blob_resource.read_sha256_metadata("public", "round-trip/foo.bin") == sha


def test_upload_overwrite_preserves_metadata(blob_resource: BlobStorageResource) -> None:
    """Re-uploading replaces the blob and writes fresh sha256 metadata."""
    first = b"first"
    second = b"second"
    first_sha = hashlib.sha256(first).hexdigest()
    second_sha = hashlib.sha256(second).hexdigest()

    blob_resource.upload("public", "overwrite/x.bin", first, sha256=first_sha)
    assert blob_resource.read_sha256_metadata("public", "overwrite/x.bin") == first_sha

    blob_resource.upload("public", "overwrite/x.bin", second, sha256=second_sha)
    assert blob_resource.read_sha256_metadata("public", "overwrite/x.bin") == second_sha


def test_read_sha256_metadata_missing_blob(blob_resource: BlobStorageResource) -> None:
    """Reading metadata for a non-existent blob returns ``None``."""
    assert blob_resource.read_sha256_metadata("public", "does-not-exist.bin") is None


def test_read_sha256_metadata_blob_without_metadata(
    blob_resource: BlobStorageResource,
    blob_container_name: str,
    azurite_connection_string: str,
) -> None:
    """A blob uploaded without sha256 metadata returns ``None``.

    Mirrors the "predates this framework" case from the resource
    docstring -- callers should treat it as re-upload.
    """
    from azure.storage.blob import BlobServiceClient

    raw_service = BlobServiceClient.from_connection_string(azurite_connection_string)
    raw_service.get_blob_client(blob_container_name, "legacy/no-meta.bin").upload_blob(
        b"legacy-content", overwrite=True
    )

    assert blob_resource.read_sha256_metadata("public", "legacy/no-meta.bin") is None


def test_list_returns_uploaded_blob_paths(blob_resource: BlobStorageResource) -> None:
    """``list`` enumerates uploaded blobs under a prefix."""
    blob_resource.upload("public", "prefix/a.bin", b"A", sha256=hashlib.sha256(b"A").hexdigest())
    blob_resource.upload("public", "prefix/b.bin", b"B", sha256=hashlib.sha256(b"B").hexdigest())
    blob_resource.upload("public", "other/c.bin", b"C", sha256=hashlib.sha256(b"C").hexdigest())

    paths = sorted(blob_resource.list("public", "prefix"))

    assert paths == ["prefix/a.bin", "prefix/b.bin"]


def test_download_returns_uploaded_bytes(blob_resource: BlobStorageResource) -> None:
    """``download`` returns the same bytes that were uploaded."""
    payload = b"download-me" * 100
    sha = hashlib.sha256(payload).hexdigest()
    blob_resource.upload("public", "download/x.bin", payload, sha256=sha)

    assert blob_resource.download("public", "download/x.bin") == payload


def test_stream_returns_chunked_io_against_real_sdk(
    blob_resource: BlobStorageResource,
) -> None:
    """End-to-end coverage for #241: ``stream()`` returns a working
    ``IO[bytes]`` over the SDK's chunked downloader.

    Pre-fix this method tried to pass ``max_chunk_get_size`` per-call
    to ``download_blob(...)`` -- the SDK pipeline forwarded the unknown
    kwarg through to ``requests.Session.request`` and raised
    ``TypeError`` (caught here when the unit tests' MagicMock could
    not).  Bounded ``read(n)`` consumes the full payload incrementally.
    """
    payload = b"stream-me-please" * 1024  # ~16 KiB; real-SDK round-trip
    sha = hashlib.sha256(payload).hexdigest()
    blob_resource.upload("public", "stream/x.bin", payload, sha256=sha)

    out = bytearray()
    with blob_resource.stream("public", "stream/x.bin") as fp:
        while True:
            chunk = fp.read(4096)
            if not chunk:
                break
            out += chunk
    assert bytes(out) == payload


def test_stream_unbounded_read_raises_against_real_sdk(
    blob_resource: BlobStorageResource,
) -> None:
    """End-to-end pin for #248 review item 3: ``read()`` (no size)
    must raise even against the real SDK -- the streaming bound the
    adapter advertises is only meaningful if unbounded reads can't
    silently bypass it.
    """
    import io as _io

    payload = b"x" * 64
    blob_resource.upload(
        "public", "stream/unbounded.bin", payload, sha256=hashlib.sha256(payload).hexdigest()
    )

    with blob_resource.stream("public", "stream/unbounded.bin") as fp:
        with pytest.raises(_io.UnsupportedOperation, match="Unbounded read"):
            fp.read()
        # Bounded reads continue to work after the rejection.
        assert fp.read(64) == payload


def test_download_to_path_writes_file_to_disk(
    blob_resource: BlobStorageResource, tmp_path: Path
) -> None:
    """End-to-end coverage for #241: ``download_to_path`` writes the
    blob to a local file via the SDK's optimized streaming path.

    Pre-fix the same per-call-kwarg bug that broke ``download()`` and
    ``stream()`` would have broken this method too; pin it here.
    """
    payload = b"to-disk-payload" * 256  # ~4 KiB
    sha = hashlib.sha256(payload).hexdigest()
    blob_resource.upload("public", "to-disk/x.bin", payload, sha256=sha)

    dest = tmp_path / "downloaded.bin"
    blob_resource.download_to_path("public", "to-disk/x.bin", dest)
    assert dest.read_bytes() == payload


def test_exists_reports_true_after_upload(blob_resource: BlobStorageResource) -> None:
    """``exists`` flips from False to True after an upload."""
    assert blob_resource.exists("public", "exists/probe.bin") is False
    blob_resource.upload(
        "public", "exists/probe.bin", b"x", sha256=hashlib.sha256(b"x").hexdigest()
    )
    assert blob_resource.exists("public", "exists/probe.bin") is True
