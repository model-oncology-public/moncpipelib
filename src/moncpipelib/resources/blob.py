"""Azure Blob Storage resource for the universal ingest boundary.

This module provides :class:`BlobStorageResource`, a Dagster
:class:`~dagster.ConfigurableResource` that backs the ingest landing
boundary.  Every external source lands in a sensitivity-scoped container
before any downstream asset reads it.

Security / compliance context:

- Workload identity federation only -- no shared keys, no connection strings.
  Credentials flow through :class:`~azure.identity.DefaultAzureCredential`.
- SHA-256 per object is stored as blob metadata (``x-ms-meta-sha256``) on
  upload, enabling cheap HEAD-only idempotency checks.
- Container selection is driven by sensitivity class; confidential and PHI
  containers are independent of the public container and may be left
  unconfigured until their use-case exists.
- Supports HIPAA 164.312(b) (audit controls via per-object ``IngestResult``
  events) and 164.312(c)(1) (integrity via sha256).
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal, cast

from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dagster import ConfigurableResource
from pydantic import PrivateAttr

if TYPE_CHECKING:
    from collections.abc import Iterator

    from azure.storage.blob import BlobProperties, ContainerClient
    from dagster import InitResourceContext


Sensitivity = Literal["public", "confidential", "phi"]


_DEFAULT_MAX_CHUNK_GET_SIZE: int = 8 * 1024 * 1024
"""Default per-chunk download size (8 MiB).

Matches the upload-side block size landed in #239.  Bounded enough for
tight pod limits, large enough to amortize per-call HTTP overhead.
"""


class _ChunkedBlobReader(io.RawIOBase):
    """Forward-only ``IO[bytes]`` adapter over Azure SDK's ``StorageStreamDownloader``.

    Implements the :class:`io.RawIOBase` contract so ``read(n)`` /
    ``readinto(buf)`` callers see incremental bytes without the SDK ever
    materializing the whole blob.  Memory footprint is bounded by the
    SDK's ``max_chunk_get_size`` (passed at ``download_blob`` call time).

    **Forward-only**: ``seekable()`` is ``False``.  Consumers that need
    seek (notably :class:`zipfile.ZipFile`, which reads the central
    directory at end-of-file) must use
    :meth:`BlobStorageResource.download_to_path` to materialize the
    blob to disk first, then open the local file.

    **Bounded reads only.**  ``readall()`` (and therefore ``read()``
    with no size argument) raise :class:`io.UnsupportedOperation` --
    an unbounded read would materialize the full blob and silently
    negate the streaming bound this adapter exists to provide.
    Consumers must call ``read(n)`` with a finite ``n`` or
    ``readinto(buffer)``.

    Use as a context manager (``with resource.stream(...) as fp:``) so
    the underlying HTTP response closes deterministically; the SDK
    response otherwise releases on garbage collection only.
    """

    def __init__(self, downloader: Any) -> None:
        super().__init__()
        self._downloader = downloader
        # ``StorageStreamDownloader.chunks()`` yields ``bytes`` objects of
        # the SDK-configured chunk size.  We pull one chunk at a time
        # and serve from it via an offset cursor; this avoids re-slicing
        # the chunk on every ``readinto`` (each slice would allocate a
        # near-chunk-sized bytes object).
        self._chunks = iter(downloader.chunks())
        self._buf: bytes = b""
        self._offset: int = 0
        self._exhausted = False

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        """Fill ``buffer`` with the next bytes from the underlying stream.

        Returns the number of bytes written; 0 indicates EOF.  Pulls at
        most one new SDK chunk per call -- consumers asking for more
        than one chunk's worth of bytes may need to call ``readinto``
        repeatedly.
        """
        if self.closed:
            raise ValueError("readinto on closed _ChunkedBlobReader")

        if self._offset >= len(self._buf) and not self._exhausted:
            try:
                self._buf = next(self._chunks)
                self._offset = 0
            except StopIteration:
                self._exhausted = True
                return 0

        remaining = len(self._buf) - self._offset
        if remaining == 0:
            return 0

        n = min(len(buffer), remaining)
        buffer[:n] = self._buf[self._offset : self._offset + n]
        self._offset += n
        return n

    def readall(self) -> bytes:
        """Refuse unbounded reads.

        ``RawIOBase.read(-1)`` (and ``read()`` with no argument) call
        ``readall``, which would loop ``readinto`` accumulating into a
        single ``bytes`` object -- materializing the full blob and
        silently negating the streaming bound that motivated #241.
        Consumers must call ``read(n)`` with a finite ``n`` or
        ``readinto(buffer)`` so the chunk-by-chunk consumption is
        explicit at the call site.
        """
        raise io.UnsupportedOperation(
            "Unbounded read on _ChunkedBlobReader would materialize the full "
            "blob and defeat the streaming bound; use read(n) with a finite "
            "n or readinto(buffer).  For 'I want the whole blob on disk', "
            "use BlobStorageResource.download_to_path() instead."
        )

    def close(self) -> None:
        if self.closed:
            return
        # Deterministically release the underlying HTTP response so a
        # consumer that forgets to ``with``-wrap the reader does not
        # leak a connection until GC.  ``StorageStreamDownloader`` does
        # not expose a public ``close()`` on this version of the SDK
        # (azure-storage-blob>=12.23), but the pipeline response is
        # held at ``_response``; close it defensively.  If the
        # attribute path drifts in a future SDK version, drop the
        # reference and let GC reclaim it.
        downloader = self._downloader
        response = getattr(downloader, "_response", None)
        if response is not None:
            close = getattr(response, "close", None)
            if callable(close):
                # Best-effort release: SDK version drift could change
                # what `close()` raises; the worst case here is the same
                # GC-only behavior the previous implementation had.
                with contextlib.suppress(Exception):
                    close()
        self._chunks = iter(())
        self._buf = b""
        self._offset = 0
        self._exhausted = True
        self._downloader = None
        super().close()


class BlobStorageResource(ConfigurableResource):
    """Dagster resource for reading / writing ingest blobs in ADLS Gen2.

    Container selection is resolved from the contract's ``sensitivity``
    field at call time.  Missing-sensitivity lookups raise ``ValueError``
    the first time a caller asks for a container that was not configured
    -- this surfaces misconfiguration before any network I/O happens.

    Attributes:
        storage_account: Storage account short name (no
            ``.blob.core.windows.net`` suffix).
        container_public: Container name for ``sensitivity: public``
            data.  Required in every environment that runs ingests.
        container_confidential: Container for ``sensitivity: confidential``.
            Leave ``None`` until the first confidential contract lands.
        container_phi: Container for ``sensitivity: phi``.  Leave ``None``
            until the PHI container has been provisioned with CMK and WORM
            retention.
    """

    storage_account: str
    container_public: str | None = None
    container_confidential: str | None = None
    container_phi: str | None = None
    max_chunk_get_size: int = _DEFAULT_MAX_CHUNK_GET_SIZE

    _credential: DefaultAzureCredential = PrivateAttr()
    _service_client: BlobServiceClient = PrivateAttr()

    def setup_for_execution(self, context: InitResourceContext) -> None:  # noqa: ARG002
        """Instantiate the credential + service client once per run.

        ``max_chunk_get_size`` (and ``max_single_get_size``, set to the
        same value) are passed at client-construction time -- they are
        ``StorageConfiguration`` knobs popped from kwargs by
        :class:`BlobServiceClient`'s base class, not per-download
        arguments.  Passing them per-call would forward through to
        ``requests.Session.request`` and raise ``TypeError`` (caught
        by the integration tests against Azurite).

        Setting both knobs to the same value gives a uniform streaming
        bound across blob sizes: a blob smaller than
        ``max_single_get_size`` would otherwise download as a single
        GET with a buffer scaled to the file rather than the chunk.
        """
        self._credential = DefaultAzureCredential()
        self._service_client = BlobServiceClient(
            account_url=f"https://{self.storage_account}.blob.core.windows.net",
            credential=self._credential,
            max_chunk_get_size=self.max_chunk_get_size,
            max_single_get_size=self.max_chunk_get_size,
        )

    def _container_name(self, sensitivity: Sensitivity) -> str:
        """Resolve the configured container for ``sensitivity``.

        Raises ``ValueError`` when the matching ``container_*`` field is
        unset.  This protects against landing sensitive data in the wrong
        container simply because an env var was missing.
        """
        name = {
            "public": self.container_public,
            "confidential": self.container_confidential,
            "phi": self.container_phi,
        }.get(sensitivity)
        if name is None:
            raise ValueError(
                f"No container configured for sensitivity={sensitivity!r}. "
                f"Set container_{sensitivity} on BlobStorageResource."
            )
        return name

    def _container_for(self, sensitivity: Sensitivity) -> ContainerClient:
        return self._service_client.get_container_client(self._container_name(sensitivity))

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def upload(
        self,
        sensitivity: Sensitivity,
        path: str,
        data: bytes | IO[bytes],
        sha256: str,
    ) -> None:
        """Upload ``data`` to the container resolved from ``sensitivity``.

        Sets ``x-ms-meta-sha256`` so subsequent idempotency checks can
        use HEAD-only reads via :meth:`read_sha256_metadata`.  Overwrites
        any existing blob at ``path``.

        Note: the metadata key casing written here is not guaranteed to
        round-trip.  Azure Storage normalizes ``x-ms-meta-*`` headers and
        may return them with different casing (``Sha256``, ``SHA256``,
        etc.).  Readers must be case-insensitive; see
        :meth:`read_sha256_metadata`.
        """
        blob_client = self._container_for(sensitivity).get_blob_client(path)
        blob_client.upload_blob(
            data,
            overwrite=True,
            metadata={"sha256": sha256},
        )

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def download(self, sensitivity: Sensitivity, path: str) -> bytes:
        """Read the full contents of a blob into memory.

        Use only for blobs known to be small (a few MB at most).  For
        larger payloads, use :meth:`stream` (forward-only file-like) or
        :meth:`download_to_path` (write directly to local disk) -- both
        keep peak memory bounded by ``max_chunk_get_size`` regardless of
        blob size.
        """
        blob_client = self._container_for(sensitivity).get_blob_client(path)
        # Chunk size is configured at the BlobServiceClient level (see
        # setup_for_execution); not a per-call kwarg.
        return blob_client.download_blob().readall()

    def stream(self, sensitivity: Sensitivity, path: str) -> IO[bytes]:
        """Return a forward-only ``IO[bytes]`` over the blob's contents.

        The returned reader pulls one SDK chunk at a time
        (``max_chunk_get_size``, default 8 MiB), so peak memory is
        bounded by the chunk size regardless of blob size.  This is the
        right primitive for piping a blob into a streaming JSON parser,
        a streaming archive reader (excluding ``zipfile`` -- see below),
        or any consumer that reads forward and never seeks.

        **Forward-only.**  ``seekable()`` returns ``False``.  Consumers
        that seek -- :class:`zipfile.ZipFile` (reads the central
        directory at end-of-file) and most binary parsers that mmap or
        random-access the input -- must use :meth:`download_to_path` to
        write the blob to local disk first, then open the on-disk file.

        **Bounded reads only.**  ``read()`` / ``readall()`` (no size
        argument) raise :class:`io.UnsupportedOperation` -- an
        unbounded read would materialize the full blob and silently
        negate the streaming bound this method exists to provide.
        Consumers must use ``read(n)`` with a finite ``n`` or
        ``readinto(buffer)``.  For "I want the whole blob in memory"
        use :meth:`download` (small payloads only); for "I want it on
        disk" use :meth:`download_to_path`.

        Wrap the returned reader in a ``with`` block (or close it
        explicitly) so the underlying HTTP response is released
        deterministically; otherwise it releases on garbage collection
        only and a busy pod can pressure the connection pool.
        """
        blob_client = self._container_for(sensitivity).get_blob_client(path)
        # Chunk size is configured at the BlobServiceClient level (see
        # setup_for_execution); not a per-call kwarg.
        downloader = blob_client.download_blob()
        # _ChunkedBlobReader is structurally IO[bytes] (RawIOBase implements
        # the read / readable / close protocol), but typing.IO[bytes] is a
        # separate hierarchy from io.RawIOBase, so the cast is a typing
        # boundary, not a runtime concern.
        return cast("IO[bytes]", _ChunkedBlobReader(downloader))

    def download_to_path(
        self,
        sensitivity: Sensitivity,
        src: str,
        dest: Path | str,
    ) -> None:
        """Stream-download a blob to a local file.

        Uses the SDK's ``StorageStreamDownloader.readinto`` -- the
        optimized streaming-to-disk path that avoids per-chunk Python
        heap allocations entirely.  Peak memory is bounded by
        ``max_chunk_get_size`` regardless of blob size.

        On failure the destination file is unlinked so callers do not
        observe a half-written file.

        Args:
            sensitivity: Container sensitivity class.
            src: Source blob path within the resolved container.
            dest: Local filesystem path to write to.  Existing files at
                ``dest`` are overwritten.
        """
        dest_path = Path(dest)
        blob_client = self._container_for(sensitivity).get_blob_client(src)
        try:
            with dest_path.open("wb") as fp:
                # Chunk size is configured at the BlobServiceClient level
                # (see setup_for_execution); not a per-call kwarg.
                blob_client.download_blob().readinto(fp)
        except Exception:
            # Don't leave a partial file behind; the caller will retry or
            # surface the failure, and a half-written file would mask the
            # error on a follow-up read.
            dest_path.unlink(missing_ok=True)
            raise

    def iter_list(self, sensitivity: Sensitivity, prefix: str) -> Iterator[str]:
        """Yield blob paths under ``prefix`` in the sensitivity container.

        The Azure SDK's ``container.list_blobs(...)`` is itself paginated
        and lazy; this method preserves that laziness all the way to
        the caller.  Use :meth:`iter_list` for any consumer that does
        ``for path in ...`` or ``next(...)`` on the result -- a partition
        prefix with 100k+ files (e.g. UMLS Metathesaurus once unpacked)
        would otherwise materialize ~10 MiB of blob-name strings on the
        Python heap before the consumer sees the first name.

        :meth:`list` is preserved as a thin wrapper for callers that
        explicitly need a fully-materialized ``list[str]`` (e.g. tests
        asserting on the full set, or when the caller plans to iterate
        the result more than once).

        Migration 012 Phase E (#246).
        """
        container = self._container_for(sensitivity)
        for blob in container.list_blobs(name_starts_with=prefix):
            yield blob.name

    def list(self, sensitivity: Sensitivity, prefix: str) -> list[str]:
        """List blob paths under ``prefix`` in the sensitivity container.

        Returns a fully-materialized ``list[str]``.  Callers that
        iterate the result once should prefer :meth:`iter_list`, which
        yields lazily and is bounded by the SDK's pagination cursor
        rather than the total prefix's file count.
        """
        return list(self.iter_list(sensitivity, prefix))

    def exists(self, sensitivity: Sensitivity, path: str) -> bool:
        """Return ``True`` iff a blob exists at ``path``."""
        return self._container_for(sensitivity).get_blob_client(path).exists()

    def get_properties(self, sensitivity: Sensitivity, path: str) -> BlobProperties:
        """Return the SDK's ``BlobProperties`` for ``path``."""
        return self._container_for(sensitivity).get_blob_client(path).get_blob_properties()

    def read_sha256_metadata(self, sensitivity: Sensitivity, path: str) -> str | None:
        """Return the blob's ``x-ms-meta-sha256`` value or ``None``.

        Returns ``None`` when the blob does not exist **or** when the
        metadata header is absent.  Either state is treated by callers as
        "re-upload": a blob predating this framework has no sha256 header
        and must be replaced rather than skipped.
        """
        blob_client = self._container_for(sensitivity).get_blob_client(path)
        try:
            props = blob_client.get_blob_properties()
        except ResourceNotFoundError:
            return None
        return self._metadata_get_ci(props.metadata, "sha256")

    @staticmethod
    def _metadata_get_ci(metadata: dict[str, str] | None, key: str) -> str | None:
        # Azure Storage does not guarantee write/read casing parity for
        # x-ms-meta-* headers; observed in production as PascalCase
        # (``Sha256``) despite lowercase writes. Match case-insensitively.
        if not metadata:
            return None
        target = key.lower()
        for k, v in metadata.items():
            if k.lower() == target:
                return v
        return None
