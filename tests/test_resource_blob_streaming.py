"""Streaming-read tests for BlobStorageResource (#241 / Migration 012 Phase A).

Pins the contract added in #241:

- ``stream()`` returns a forward-only ``IO[bytes]`` adapter
  (``_ChunkedBlobReader``) over the SDK's chunked downloader -- not a
  fully-buffered ``BytesIO``.
- ``download_to_path()`` writes a blob to local disk via
  ``StorageStreamDownloader.readinto(fp)`` and unlinks the destination
  on mid-stream failure.
- ``max_chunk_get_size`` is propagated from the resource constructor
  to the SDK at ``BlobServiceClient`` construction time (it's a
  ``StorageConfiguration`` knob, not a per-call kwarg -- passing it
  per-call sends it through to ``requests.Session.request`` and
  raises ``TypeError``).
- Peak Python heap during a ``stream()`` consumption stays bounded by
  the chunk size regardless of blob size.
"""

from __future__ import annotations

import io
import tracemalloc
from pathlib import Path
from typing import IO, Any
from unittest.mock import MagicMock, patch

import pytest
from azure.core.exceptions import ResourceNotFoundError

from moncpipelib.resources.blob import (
    _DEFAULT_MAX_CHUNK_GET_SIZE,
    BlobStorageResource,
    _ChunkedBlobReader,
)

# ---------------------------------------------------------------------------
# Helpers: resource with patched SDK + chunk-yielding mock downloader
# ---------------------------------------------------------------------------


def _build_resource(
    **overrides: object,
) -> tuple[BlobStorageResource, MagicMock, MagicMock]:
    """Construct a BlobStorageResource against a patched SDK.

    Returns ``(resource, service_instance, service_cls)``.  The class
    mock is exposed so tests can assert how
    :meth:`BlobStorageResource.setup_for_execution` constructed the
    underlying ``BlobServiceClient`` (notably the
    ``max_chunk_get_size`` / ``max_single_get_size`` kwargs that
    configure ``StorageConfiguration``).
    """
    defaults: dict[str, object] = {
        "storage_account": "examplestorageacct",
        "container_public": "landing-reference",
    }
    defaults.update(overrides)
    with (
        patch("moncpipelib.resources.blob.DefaultAzureCredential") as _cred,
        patch("moncpipelib.resources.blob.BlobServiceClient") as mock_service_cls,
    ):
        del _cred
        service = MagicMock(name="BlobServiceClient")
        mock_service_cls.return_value = service
        resource = BlobStorageResource(**defaults)  # type: ignore[arg-type]
        resource.setup_for_execution(MagicMock(name="InitResourceContext"))
    return resource, service, mock_service_cls


def _mock_downloader(chunks: list[bytes]) -> MagicMock:
    """Build a mock StorageStreamDownloader yielding ``chunks`` from .chunks().

    Also wires ``readinto(fp)`` to drain the chunk list into ``fp`` in
    order, mirroring the SDK's behavior for ``download_to_path``.  The
    same downloader can serve either flow path; tests pick which they
    exercise.
    """
    downloader = MagicMock(name="StorageStreamDownloader")
    downloader.chunks.return_value = iter(chunks)

    def _readinto(fp: IO[bytes]) -> int:
        total = 0
        for c in chunks:
            fp.write(c)
            total += len(c)
        return total

    downloader.readinto.side_effect = _readinto
    downloader.readall.return_value = b"".join(chunks)
    return downloader


# ---------------------------------------------------------------------------
# _ChunkedBlobReader adapter
# ---------------------------------------------------------------------------


def _drain_bounded(reader: _ChunkedBlobReader, chunk_size: int = 256) -> bytes:
    """Drain a reader via bounded ``read(n)`` calls.

    Pins that the unbounded-read path is the right idiom: tests that
    want the full payload still work, they just call ``read(n)`` in a
    loop instead of ``read()`` (which now raises by design -- see
    :meth:`_ChunkedBlobReader.readall`).
    """
    out = b""
    while True:
        b = reader.read(chunk_size)
        if not b:
            break
        out += b
    return out


def test_chunked_reader_serves_bytes_in_order() -> None:
    """Reading concatenates the SDK chunks in the order they arrive."""
    downloader = _mock_downloader([b"abc", b"def", b"ghi"])
    reader = _ChunkedBlobReader(downloader)
    assert reader.readable()
    out = _drain_bounded(reader)
    assert out == b"abcdefghi"


def test_chunked_reader_pulls_at_most_one_chunk_per_readinto() -> None:
    """Pins the streaming bound: a single ``readinto(buf)`` call advances
    by at most one SDK chunk regardless of ``len(buf)``.

    Together with the ``size = min(len(buf), len(leftover))`` arithmetic
    in the adapter, this means the adapter never holds more than one
    chunk's worth of bytes in memory.
    """
    chunks = [b"x" * 1024, b"y" * 1024, b"z" * 1024]
    reader = _ChunkedBlobReader(_mock_downloader(chunks))
    buf = bytearray(8192)
    n = reader.readinto(buf)
    assert n == 1024  # served one chunk only, not the requested 8192
    assert bytes(buf[:n]) == b"x" * 1024


def test_chunked_reader_reports_eof_after_drain() -> None:
    reader = _ChunkedBlobReader(_mock_downloader([b"only"]))
    assert _drain_bounded(reader) == b"only"
    assert reader.read(16) == b""  # subsequent reads return EOF


def test_chunked_reader_close_is_idempotent() -> None:
    reader = _ChunkedBlobReader(_mock_downloader([b"one"]))
    reader.close()
    reader.close()  # no-op
    assert reader.closed
    with pytest.raises(ValueError, match="closed"):
        reader.readinto(bytearray(4))


def test_chunked_reader_is_forward_only() -> None:
    """``zipfile.ZipFile`` consumers must use download_to_path; pin that
    by asserting ``seekable()`` is False (RawIOBase default)."""
    reader = _ChunkedBlobReader(_mock_downloader([b"x"]))
    assert reader.seekable() is False


def test_chunked_reader_unbounded_read_raises() -> None:
    """``read()`` / ``readall()`` (no size) must raise rather than
    materializing the full blob.  Pre-fix the inherited
    ``RawIOBase.readall`` would loop ``readinto`` accumulating into a
    single ``bytes`` object -- silently negating the streaming bound.

    Bounded ``read(n)`` continues to work; this test pins both halves.
    """
    reader = _ChunkedBlobReader(_mock_downloader([b"hello"]))
    with pytest.raises(io.UnsupportedOperation, match="Unbounded read"):
        reader.read()
    with pytest.raises(io.UnsupportedOperation, match="Unbounded read"):
        reader.read(-1)
    with pytest.raises(io.UnsupportedOperation, match="Unbounded read"):
        reader.readall()
    # Bounded read still works.
    assert reader.read(16) == b"hello"


def test_chunked_reader_close_releases_sdk_response() -> None:
    """``close()`` must call ``close()`` on the underlying SDK
    response so a consumer that fails to ``with``-wrap the reader does
    not leak an HTTP connection until GC.

    The Azure SDK's ``StorageStreamDownloader`` does not expose a
    public close on the version pinned (azure-storage-blob>=12.23) but
    holds the pipeline response at ``_response``.  The adapter closes
    it defensively; if the attribute path drifts in a future SDK
    version, this test will surface the regression.
    """
    chunks = [b"x"]
    downloader = _mock_downloader(chunks)
    response = MagicMock(name="PipelineResponse")
    downloader._response = response

    reader = _ChunkedBlobReader(downloader)
    reader.close()

    response.close.assert_called_once_with()


def test_chunked_reader_close_tolerates_missing_sdk_response() -> None:
    """If a future SDK version drops ``_response`` entirely, the
    adapter must still close cleanly (drop the reference; let GC
    reclaim).  Otherwise an SDK upgrade silently breaks every
    ``with`` block in the codebase.

    Uses a real class (not MagicMock) so ``getattr(downloader,
    "_response", None)`` returns the sentinel rather than an
    auto-created child mock.
    """

    class _MinimalDownloader:
        # No _response attribute; mimic a hypothetical future SDK.
        def __init__(self, chunks: list[bytes]) -> None:
            self._chunk_list = chunks

        def chunks(self) -> Any:
            return iter(self._chunk_list)

    reader = _ChunkedBlobReader(_MinimalDownloader([b"x"]))
    reader.close()  # must not raise
    assert reader.closed


# ---------------------------------------------------------------------------
# stream() integration
# ---------------------------------------------------------------------------


def test_stream_returns_chunked_reader_wired_to_downloader() -> None:
    resource, service, _service_cls = _build_resource()
    blob_client = service.get_container_client.return_value.get_blob_client.return_value
    blob_client.download_blob.return_value = _mock_downloader([b"hello ", b"world"])

    fp = resource.stream("public", "some/path.bin")
    # Bounded reads only -- ``read()`` (no size) raises by design; pin
    # that elsewhere and use ``read(n)`` here.
    out = b""
    while True:
        b = fp.read(64)
        if not b:
            break
        out += b
    assert out == b"hello world"
    # ``download_blob`` is called with no kwargs; the chunk size is
    # configured at the BlobServiceClient level (see
    # test_max_chunk_get_size_propagated_to_service_client_constructor).
    blob_client.download_blob.assert_called_once_with()


def test_max_chunk_get_size_propagated_to_service_client_constructor() -> None:
    """``max_chunk_get_size`` is a ``StorageConfiguration`` knob popped
    from kwargs by ``BlobServiceClient.__init__`` -- not a per-call
    argument to ``download_blob``.  Passing it per-call would forward
    through to ``requests.Session.request`` and raise ``TypeError``
    (caught by the integration tests against Azurite).

    Also assert ``max_single_get_size`` is set to the same value: a
    blob smaller than the SDK's default (32 MiB) would otherwise
    download as a single GET sized to the file rather than the chunk,
    breaking the streaming bound for small-but-not-tiny payloads.
    """
    custom_size = 16 * 1024 * 1024
    _resource, _service, service_cls = _build_resource(max_chunk_get_size=custom_size)

    service_cls.assert_called_once()
    kwargs = service_cls.call_args.kwargs
    assert kwargs["max_chunk_get_size"] == custom_size
    assert kwargs["max_single_get_size"] == custom_size


def test_default_max_chunk_get_size_propagated() -> None:
    """When the resource is constructed without an override, the
    default 8 MiB chunk size is forwarded to the BlobServiceClient."""
    _resource, _service, service_cls = _build_resource()

    kwargs = service_cls.call_args.kwargs
    assert kwargs["max_chunk_get_size"] == _DEFAULT_MAX_CHUNK_GET_SIZE
    assert kwargs["max_single_get_size"] == _DEFAULT_MAX_CHUNK_GET_SIZE


def test_stream_64mib_blob_stays_bounded() -> None:
    """tracemalloc acceptance bar: peak Python heap during a 64 MiB
    ``stream()`` consumption stays under the chunk-size bound regardless
    of total blob size.

    Pre-#241 this read materialized the full 64 MiB in BytesIO; peak
    would track blob size.  With the chunked adapter, peak tracks the
    8 MiB chunk.
    """
    member_size = 64 * 1024 * 1024
    chunk_size = 8 * 1024 * 1024
    threshold = 32 * 1024 * 1024
    chunk = b"a" * chunk_size  # one shared bytes object across chunks

    # Pre-allocate the chunk list (each entry is a reference to the same
    # bytes object) to keep test setup out of the measured baseline.
    chunks_list = [chunk] * (member_size // chunk_size)

    resource, service, _service_cls = _build_resource()
    blob_client = service.get_container_client.return_value.get_blob_client.return_value
    blob_client.download_blob.return_value = _mock_downloader(chunks_list)

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        with resource.stream("public", "some/path.bin") as fp:
            total = 0
            buf = bytearray(chunk_size)
            while True:
                n = fp.readinto(buf)
                if n == 0:
                    break
                total += n
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert total == member_size
    assert peak <= threshold, (
        f"peak Python heap was {peak / 1024 / 1024:.1f} MiB during a "
        f"{member_size / 1024 / 1024:.0f} MiB stream() -- streaming "
        f"regression?  Threshold: {threshold / 1024 / 1024:.0f} MiB."
    )


# ---------------------------------------------------------------------------
# download_to_path()
# ---------------------------------------------------------------------------


def test_download_to_path_round_trip_equality(tmp_path: Path) -> None:
    """``download_to_path`` writes the full blob payload to disk."""
    resource, service, _service_cls = _build_resource()
    blob_client = service.get_container_client.return_value.get_blob_client.return_value
    payload = b"hello world payload"
    blob_client.download_blob.return_value = _mock_downloader([payload[:5], payload[5:]])

    dest = tmp_path / "dest.bin"
    resource.download_to_path("public", "src/path.bin", dest)

    assert dest.read_bytes() == payload
    # ``download_blob`` is called with no kwargs; chunk size is
    # configured at BlobServiceClient construction time.
    blob_client.download_blob.assert_called_once_with()


def test_download_to_path_unlinks_partial_file_on_failure(tmp_path: Path) -> None:
    """A mid-stream exception must not leave a half-written file behind --
    the next read would otherwise see corrupt content masquerading as
    successful state.
    """
    resource, service, _service_cls = _build_resource()
    blob_client = service.get_container_client.return_value.get_blob_client.return_value
    downloader = MagicMock(name="StorageStreamDownloader")
    downloader.readinto.side_effect = OSError("disk full")
    blob_client.download_blob.return_value = downloader

    dest = tmp_path / "partial.bin"
    with pytest.raises(OSError, match="disk full"):
        resource.download_to_path("public", "src/path.bin", dest)
    assert not dest.exists()


def test_download_to_path_overwrites_existing(tmp_path: Path) -> None:
    """Existing files at ``dest`` are replaced; we open in 'wb'."""
    resource, service, _service_cls = _build_resource()
    blob_client = service.get_container_client.return_value.get_blob_client.return_value
    blob_client.download_blob.return_value = _mock_downloader([b"new content"])

    dest = tmp_path / "exists.bin"
    dest.write_bytes(b"old content -- much longer than new")
    resource.download_to_path("public", "src/path.bin", dest)
    assert dest.read_bytes() == b"new content"


def test_download_to_path_propagates_resource_not_found(tmp_path: Path) -> None:
    resource, service, _service_cls = _build_resource()
    blob_client = service.get_container_client.return_value.get_blob_client.return_value
    blob_client.download_blob.side_effect = ResourceNotFoundError("not found")

    dest = tmp_path / "missing.bin"
    with pytest.raises(ResourceNotFoundError):
        resource.download_to_path("public", "src/path.bin", dest)
    # Cleanup safety: nothing was written, but the empty file may have
    # been opened+closed before the exception.  The unlink in the
    # except branch handles either state.
    assert not dest.exists()


# ---------------------------------------------------------------------------
# download() docstring contract regression
# ---------------------------------------------------------------------------


def test_download_calls_download_blob_with_no_kwargs() -> None:
    """``download()`` calls ``download_blob()`` with no kwargs; chunk
    size is configured at the BlobServiceClient level.

    Pre-fix this method passed ``max_chunk_get_size`` per-call, which
    flowed through the SDK pipeline to ``requests.Session.request``
    and raised ``TypeError`` against real Azurite.  Pinning the
    no-kwargs shape catches a regression at unit-test time without
    needing the integration container.
    """
    resource, _service, service_cls = _build_resource(max_chunk_get_size=4 * 1024 * 1024)
    blob_client = (
        resource._container_for("public").get_blob_client.return_value  # type: ignore[attr-defined]
    )
    blob_client.download_blob.return_value = _mock_downloader([b"small"])

    out = resource.download("public", "some/small/blob")
    assert out == b"small"
    blob_client.download_blob.assert_called_once_with()
    # The chunk size landed on the constructor instead.
    assert service_cls.call_args.kwargs["max_chunk_get_size"] == 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# Smoke: stream() works as a context manager
# ---------------------------------------------------------------------------


def test_stream_usable_as_context_manager() -> None:
    """RawIOBase is a context manager (closes on __exit__); pin that
    callers can use ``with resource.stream(...) as fp:`` for prompt
    HTTP-response release.
    """
    resource, service, _service_cls = _build_resource()
    blob_client = service.get_container_client.return_value.get_blob_client.return_value
    blob_client.download_blob.return_value = _mock_downloader([b"x", b"y"])

    with resource.stream("public", "p") as fp:
        out = fp.read(64) + fp.read(64)
        assert isinstance(fp, io.RawIOBase)
    assert out == b"xy"
    assert fp.closed
