"""Forward-only chunked ``IO[bytes]`` adapter over the Azure SDK downloader.

Shared by :class:`~moncpipelib.resources.blob.BlobStorageResource` (our
WIF single-account landing boundary) and
:class:`~moncpipelib.resources.foreign_blob.ForeignBlobSource` (the
SP-credentialed cross-tenant read source, #436).  Both need the same
streaming-read primitive -- pull one SDK chunk at a time so peak memory
is bounded by ``max_chunk_get_size`` regardless of blob size (#241) --
so the adapter lives here once rather than being duplicated per resource.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any

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
    directory at end-of-file) must materialize the blob to disk first
    (e.g. via ``download_to_path``), then open the local file.

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
            "n or readinto(buffer)."
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
