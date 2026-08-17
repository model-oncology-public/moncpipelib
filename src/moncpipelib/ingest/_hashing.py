"""Hash-while-writing tempfile helper shared by ingest write paths.

The ingest layer computes sha256 + size for streamed content in the
same pass as the write, never by re-reading the bytes (#239).  That
idiom now has three consumers -- extracted archive members
(``patterns/_extract.py``), the per-partition manifest
(``dispatcher._write_manifest_to_tempfile``), and the ``api_crawl``
NDJSON assembly (#415) -- so the tempfile + hasher plumbing lives here
once.

Usage::

    with hashing_tempfile(suffix=".ndjson") as writer:
        for chunk in produce_chunks():
            writer.write(chunk)
        writer.close()  # flush to disk; path is now readable
        upload(writer.path, sha256=writer.sha256_hexdigest(),
               size=writer.size_bytes)
    # tempfile unlinked here, success or failure

Memory profile: peak heap is bounded by the largest single ``write``
call, not the total bytes written -- the I/O-at-Boundaries invariant
from ``CLAUDE.md``.  Cleanup: the tempfile is unlinked on context exit
even when the body raises, so a failure mid-stream does not leak
partial bytes onto the pod's ephemeral disk (see SECURITY.md,
"Transient Files on Pod-Local Disk").
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO


class HashingTempfileWriter:
    """Write-only tempfile sink that folds every write into a sha256.

    Construct via :func:`hashing_tempfile`, not directly -- the context
    manager owns tempfile creation and unlink-on-exit.

    Callers MUST :meth:`close` before reading :attr:`path` (on Windows
    the open handle would block the read; on all platforms the close
    flushes buffered bytes).  ``close`` is idempotent; the context
    manager calls it again on exit as a safety net.
    """

    def __init__(self, handle: IO[bytes], path: Path) -> None:
        self._handle = handle
        self._hasher = hashlib.sha256()
        self._size = 0
        self._closed = False
        self.path = path

    def write(self, data: bytes) -> int:
        """Write ``data`` to the tempfile and fold it into the digest."""
        self._hasher.update(data)
        written = self._handle.write(data)
        self._size += written
        return written

    def close(self) -> None:
        """Flush and close the underlying tempfile.  Idempotent."""
        if not self._closed:
            self._handle.close()
            self._closed = True

    def sha256_hexdigest(self) -> str:
        """Hex sha256 of every byte written so far."""
        return self._hasher.hexdigest()

    @property
    def size_bytes(self) -> int:
        """Total bytes written so far."""
        return self._size


@contextmanager
def hashing_tempfile(suffix: str) -> Iterator[HashingTempfileWriter]:
    """Yield a :class:`HashingTempfileWriter` backed by a fresh tempfile.

    The tempfile is created with ``delete=False`` so the writer can be
    closed and the path re-opened for reading within the context (the
    upload path needs an independent read handle).  The path is
    unlinked when the context exits -- success or failure -- so callers
    must consume the file inside the ``with`` block.

    Args:
        suffix: Tempfile suffix (e.g. ``".ndjson"``, ``".manifest.json"``);
            aids identification if a file is ever observed mid-run on
            the pod's ephemeral disk.
    """
    # ``delete=False`` because the caller needs to close the handle and
    # then read the path -- on Windows, opening the path while the
    # NamedTemporaryFile holds it would fail.  Unlinked in the finally.
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)  # noqa: SIM115
    path = Path(handle.name)
    writer = HashingTempfileWriter(handle, path)
    try:
        yield writer
    finally:
        writer.close()
        path.unlink(missing_ok=True)
