"""Bounded-stream helpers for ingest consumers.

The ingest read path (:meth:`BlobStorageResource.stream`,
:func:`read_partition_with_manifest`) hands callers a *bounded-only*
``IO[bytes]`` -- the production reader
(:class:`~moncpipelib.resources.blob._ChunkedBlobReader`) refuses
``read()`` / ``readall()`` with no size argument so an unbounded read
cannot silently materialize the whole blob and negate the streaming
bound (#241).

Most reference-file parsers, however, want the whole (small) payload in
a seekable buffer: polars / pandas ``read_csv``, ``calamine`` /
``openpyxl`` for XLSX, ``json.loads`` for a small JSON document.  Worse,
some of these (notably polars via PyO3) fall through to ``readall()``
internally even when the caller only ever called ``read(n)`` -- handing
the bounded reader straight to them panics in production (#354).

The right idiom for the small-reference-file case is therefore: drain
the bounded stream into ``bytes`` with an explicit size ceiling, then
wrap the result in :class:`io.BytesIO` for whichever parser.
:func:`drain_to_bytes` is that drain, with a ``max_bytes`` safety
ceiling so a mis-classified large blob fails loudly instead of OOMing
the pod.

For payloads that are *not* contractually small, do NOT use this helper
-- use :meth:`BlobStorageResource.download_to_path` (streams to local
disk, peak memory bounded by the chunk size) and open the on-disk file.
"""

from __future__ import annotations

from typing import IO

__all__ = ["StreamTooLargeError", "drain_to_bytes"]

_DEFAULT_MAX_BYTES: int = 16 * 1024 * 1024
"""Default ceiling (16 MiB) for :func:`drain_to_bytes`.

Comfortably above the reference files this helper exists for (the AL
XLSX is ~270 KiB; CMS reference files are typically <10 MiB) and well
below a pod memory limit.  Callers that know their bound should pass an
explicit ``max_bytes``.
"""

_DEFAULT_CHUNK_SIZE: int = 64 * 1024
"""Per-``read`` request size (64 KiB).

The underlying reader serves at most one SDK chunk per ``read`` call
regardless of the requested size, so this only bounds the *minimum*
number of ``read`` calls; it does not change peak memory.
"""


class StreamTooLargeError(ValueError):
    """A drained stream exceeded the caller's ``max_bytes`` ceiling.

    Subclasses :class:`ValueError` so existing ``except ValueError``
    handlers keep working; callers that want to distinguish a
    size-ceiling breach from other bad input can catch this directly
    (e.g. to fall back to :meth:`BlobStorageResource.download_to_path`).
    """


def drain_to_bytes(
    stream: IO[bytes],
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> bytes:
    """Read a bounded-only stream fully into memory, with a size ceiling.

    Drains ``stream`` by looping ``read(chunk_size)`` until EOF -- the
    idiom required by the bounded-only ingest reader, which refuses
    ``read()`` with no size argument.  Returns the concatenated bytes.

    Intended for *small* reference payloads only (a few MiB).  The
    ``max_bytes`` ceiling exists to turn a mis-classified large blob
    into a loud failure rather than a silent OOM: the drain stops and
    raises :class:`StreamTooLargeError` as soon as the accumulated size
    would exceed ``max_bytes`` (it never buffers more than one extra
    chunk past the limit).  For genuinely large payloads use
    :meth:`BlobStorageResource.download_to_path` and open the on-disk
    file instead.

    Typical use -- hand the result to a parser that wants seekable input::

        import io
        import polars as pl

        with read_partition_with_manifest(...) as (ref, stream, fields):
            data = io.BytesIO(drain_to_bytes(stream, max_bytes=8 * 1024 * 1024))
            df = pl.read_csv(data)

    Args:
        stream: A readable ``IO[bytes]``.  Consumed but not closed --
            the caller (or the context manager that yielded it) owns
            closing.
        max_bytes: Maximum number of bytes to accept.  Must be positive.
            Defaults to 16 MiB.
        chunk_size: Bytes requested per ``read`` call.  Must be
            positive.  Defaults to 64 KiB.  Does not affect peak memory
            (the reader serves at most one chunk per call regardless).

    Returns:
        The full stream contents as ``bytes``.

    Raises:
        StreamTooLargeError: When the stream yields more than
            ``max_bytes`` bytes.
        ValueError: When ``max_bytes`` or ``chunk_size`` is not positive.
    """
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive; got {max_bytes!r}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive; got {chunk_size!r}")

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise StreamTooLargeError(
                f"Stream exceeded max_bytes={max_bytes} (read at least "
                f"{total} bytes); refusing to materialize.  Use "
                f"BlobStorageResource.download_to_path() for large payloads, "
                f"or increase max_bytes if this size is expected."
            )
        chunks.append(chunk)
    return b"".join(chunks)
