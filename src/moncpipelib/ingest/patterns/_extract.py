"""Streaming archive extraction for ingest patterns.

Phase 2 supports nested ``zip`` archives via repeated entries in
``extract`` (e.g. ``["zip", "zip"]`` for the UMLS Metathesaurus
zip-of-zips shape) and an optional ``extract_filter`` of ``fnmatch``
globs applied to terminal (non-archive) members at every level.

Memory profile (#239 fix): the extractor streams each terminal member
chunk-by-chunk to a tempfile, computing sha256 in the same pass.  The
yielded path + hash + size let callers upload via ``IO[bytes]`` without
ever holding the full member in memory.  Peak resident memory is
``~_EXTRACT_CHUNK_BYTES`` regardless of member size, so a 241 MB FDA NDC
JSON or a multi-GB UMLS file fits comfortably in a 768 MiB pod.

The yielded :class:`pathlib.Path` is a tempfile owned by the extractor
generator: it is unlinked when the iterator advances past the yield or
when the generator is closed.  Consumers must finish reading the path
before requesting the next member.

Per ADR-1 (``docs/migrations/20260426_phase2-ingest-decisions.md``):

- Members whose names match the next-level archive format are recursed
  into REGARDLESS of ``extract_filter`` -- the filter applies only to
  terminal (non-archive) members.  This is what makes the UMLS case
  work: the outer-zip member ``2026AB-meta.nlm.zip`` does not match
  the filter ``meta/**``, but because it is a ``.zip`` we recurse into
  it before applying the filter to its contents.
- The filter is applied AFTER ``strip_extensions`` so contract authors
  write globs against the post-strip path -- the same shape downstream
  consumers see.
- Glob matching uses :func:`fnmatch.fnmatch`.  Note that ``*`` matches
  ``/`` under fnmatch semantics, so ``meta/*`` and ``meta/**`` are
  functionally identical (``**`` is recommended for author readability).
"""

from __future__ import annotations

import hashlib
import io
import tempfile
import zipfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from fnmatch import fnmatch
from pathlib import Path
from typing import IO

ExtractSource = bytes | Path | IO[bytes]
"""What ``extract_and_filter_iter`` can accept.

- ``bytes``: in-memory payload (for small downloads / tests).
- :class:`pathlib.Path`: streamed extraction from disk; preferred for
  large downloads.
- ``IO[bytes]``: a seekable binary file-like; used for nested zip
  members during recursion (``zipfile.ZipFile.open`` returns a
  seekable stream that we feed back into the extractor).
"""

ExtractedMember = tuple[str, Path, str, int]
"""``(filename, path, sha256_hex, size_bytes)`` yielded per terminal file.

- ``filename``: the in-archive path after ``strip_extensions``.
- ``path``: a tempfile owned by the extractor generator; unlinked on
  iterator advance / close.  Consumers must read it within the iteration
  step.
- ``sha256_hex``: precomputed during the streaming write so callers can
  hash-compare without re-reading the member.
- ``size_bytes``: full byte count of the extracted member.
"""

_EXTRACT_CHUNK_BYTES: int = 8 * 1024 * 1024
"""Read / write chunk size for the streaming hash-and-write pass.

8 MiB matches Azure Blob Storage's recommended block size for medium
uploads and keeps peak in-flight buffer well within typical pod limits.
"""

_DEFAULT_PAYLOAD_NAME: str = "__payload__"
"""Helper-internal sentinel emitted as the filename slot when ``extract``
is empty (no archive expansion).

Per #270, this is a contract between the helper and its non-pattern
callers (tests, future direct callers).  Pattern code paths
(:class:`~moncpipelib.ingest.patterns.http_urls.HttpUrlsPattern`,
:class:`~moncpipelib.ingest.patterns.api_resolver.ApiResolverPattern`)
IGNORE this slot for the ``extract: []`` branch and substitute a
filename derived from the precedence chain in
:mod:`moncpipelib.ingest.patterns._payload_naming` (template ->
resolver hint -> Content-Disposition -> sanitized URL basename ->
raise) before calling ``hash_compare_and_upload``.  Do not delete the
constant: it is the helper's stable yield shape for non-pattern
callers, even though no pattern code path propagates it to upload.
"""


def extract_and_filter_iter(
    source: ExtractSource,
    extract: Sequence[str],
    strip_extensions: Sequence[str],
    extract_filter: Sequence[str] = (),
) -> Iterator[ExtractedMember]:
    """Yield :data:`ExtractedMember` tuples for each terminal file extracted.

    Args:
        source: Bytes payload, file path on disk, or seekable binary
            file-like.  See :data:`ExtractSource`.
        extract: Ordered archive formats to expand, e.g. ``("zip",)`` for
            single-zip sources or ``("zip", "zip")`` for the UMLS
            zip-of-zips shape.  The N-th entry describes the format at
            depth N.  When ``extract`` is empty the source is treated
            as a single terminal file emitted under the sentinel key
            :data:`_DEFAULT_PAYLOAD_NAME` (``"__payload__"``).  Pattern
            callers (``HttpUrlsPattern``, ``ApiResolverPattern``) ignore
            this slot for the ``extract: []`` branch and substitute a
            filename via the precedence chain in
            :mod:`moncpipelib.ingest.patterns._payload_naming` (#270);
            non-pattern callers (tests, future direct callers) see the
            sentinel verbatim.
        strip_extensions: File extensions to strip from each terminal
            file's name (e.g. ``[".xls", ".xlsx"]`` turns
            ``"foo.csv.xls"`` into ``"foo.csv"``).  Stripping happens
            BEFORE filtering so authors write globs against the
            post-strip path.
        extract_filter: Optional ``fnmatch`` globs applied recursively
            to terminal members.  When non-empty, only files whose
            post-strip path matches at least one glob are yielded.

    Yields:
        :data:`ExtractedMember` per terminal file.  Inner zip members
        keep their in-archive path -- they are NOT prefixed with the
        outer archive's name.  The yielded ``path`` is a tempfile owned
        by the generator; it is unlinked when the iterator advances or
        the generator is closed.

    Raises:
        ValueError: If an entry in ``extract`` is not ``"zip"`` -- only
            zip archives are supported today.
    """
    if not extract:
        with _open_source(source) as src, _hash_stream_to_tempfile(src) as (path, sha, size):
            yield _DEFAULT_PAYLOAD_NAME, path, sha, size
        return

    fmt = extract[0]
    if fmt != "zip":
        raise ValueError(
            f"Unsupported extract format {fmt!r}. Only 'zip' is supported; "
            f"add a new format to extract_and_filter_iter when you need it."
        )

    next_extract = extract[1:]
    with zipfile.ZipFile(_as_zipfile_arg(source)) as zf:
        for name in zf.namelist():
            # Skip directory entries.
            if name.endswith("/"):
                continue

            # Recurse if the member is itself the next-level archive.
            # Per ADR-1: archives are recursed regardless of
            # extract_filter; the filter applies to terminal members only.
            if next_extract and _matches_archive(name, next_extract[0]):
                with (
                    zf.open(name) as inner_stream,
                    _hash_stream_to_tempfile(inner_stream) as (inner_path, _sha, _size),
                ):
                    # The intermediate archive's hash is discarded; only
                    # terminal members surface to callers.  Materializing
                    # to disk keeps the recursive call's source uniform
                    # (Path) and bounds peak RAM.
                    yield from extract_and_filter_iter(
                        inner_path, next_extract, strip_extensions, extract_filter
                    )
                continue

            # Terminal: strip extensions, then apply filter (if any).
            stripped = _strip_suffix(name, strip_extensions)
            if extract_filter and not _matches_any_pattern(stripped, extract_filter):
                continue
            with (
                zf.open(name) as member_stream,
                _hash_stream_to_tempfile(member_stream) as (path, sha, size),
            ):
                yield stripped, path, sha, size


@contextmanager
def _hash_stream_to_tempfile(
    stream: IO[bytes],
    chunk_size: int = _EXTRACT_CHUNK_BYTES,
) -> Iterator[tuple[Path, str, int]]:
    """Stream ``stream`` to a tempfile while hashing in a single pass.

    Yields ``(path, sha256_hex, size_bytes)``.  The tempfile is unlinked
    on context exit -- consumers must finish reading the path before
    leaving the ``with`` block.

    Peak memory is bounded by ``chunk_size``; the disk write and hash
    update happen on the same in-flight chunk so there is no parallel
    buffer.
    """
    # delete=False: we close the handle before yielding the path so
    # consumers can re-open it (Windows would otherwise refuse).  Cleanup
    # in the outer finally below.
    handle = tempfile.NamedTemporaryFile(suffix=".extract", delete=False)  # noqa: SIM115
    path = Path(handle.name)
    hasher = hashlib.sha256()
    size = 0
    try:
        try:
            for chunk in iter(lambda: stream.read(chunk_size), b""):
                hasher.update(chunk)
                handle.write(chunk)
                size += len(chunk)
        finally:
            handle.close()
        yield path, hasher.hexdigest(), size
    finally:
        path.unlink(missing_ok=True)


@contextmanager
def _open_source(source: ExtractSource) -> Iterator[IO[bytes]]:
    """Yield a binary stream for ``source`` regardless of its concrete type."""
    if isinstance(source, bytes):
        yield io.BytesIO(source)
    elif isinstance(source, Path):
        with source.open("rb") as fp:
            yield fp
    else:
        # Caller owns the lifetime of an externally-supplied IO[bytes].
        yield source


def _as_zipfile_arg(source: ExtractSource) -> Path | IO[bytes]:
    """Coerce ``source`` to something :class:`zipfile.ZipFile` accepts.

    ``ZipFile`` accepts a path or a seekable file-like; wrap ``bytes``
    in a :class:`io.BytesIO` for the in-memory case.
    """
    if isinstance(source, bytes):
        return io.BytesIO(source)
    return source


def _matches_archive(name: str, fmt: str) -> bool:
    """Return whether ``name`` is an archive of format ``fmt``.

    Currently only ``"zip"`` is supported; matching is case-insensitive
    on the trailing ``.<fmt>`` suffix.
    """
    return name.lower().endswith(f".{fmt.lower()}")


def _matches_any_pattern(name: str, patterns: Sequence[str]) -> bool:
    """Return whether ``name`` matches at least one ``fnmatch`` pattern.

    Note: ``fnmatch.fnmatch`` does not give ``/`` special treatment, so
    ``meta/*`` and ``meta/**`` are functionally identical.  Per ADR-1
    we accept this rather than switching to a path-aware matcher.
    """
    return any(fnmatch(name, pat) for pat in patterns)


def _strip_suffix(name: str, extensions: Sequence[str]) -> str:
    """Remove a single trailing extension if it matches (case-insensitive)."""
    lowered = name.lower()
    for ext in extensions:
        if lowered.endswith(ext.lower()):
            return name[: -len(ext)]
    return name
