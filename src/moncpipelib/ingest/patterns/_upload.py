"""Shared hash-compare + streaming-upload helper for ingest patterns.

Both :class:`~moncpipelib.ingest.patterns.http_urls.HttpUrlsPattern` and
:class:`~moncpipelib.ingest.patterns.api_resolver.ApiResolverPattern`
walk extracted archive members through the same idempotency dance:
compare the precomputed sha256 against the blob's
``x-ms-meta-sha256`` metadata, skip on match, otherwise stream the
member from disk into the upload.

The helper consumes the four-tuple yielded by
:func:`~moncpipelib.ingest.patterns._extract.extract_and_filter_iter`
(filename + tempfile path + sha256 + size_bytes); it never re-reads the
member to hash it (#239: hashing happens during the extractor's single
write pass).  Upload re-opens the path only when the hash mismatches,
so a skipped partition does no extra disk I/O.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from moncpipelib.ingest.types import IngestResult

if TYPE_CHECKING:
    from pathlib import Path

    from moncpipelib.resources.blob import BlobStorageResource


def hash_compare_and_upload(
    blob: BlobStorageResource,
    sensitivity: Any,
    prefix: str,
    filename: str,
    path: Path,
    sha256: str,
    size_bytes: int,
) -> IngestResult:
    """Hash-compare the extracted member against the landed blob; upload on mismatch.

    The sha256 is precomputed by the extractor's streaming write, so this
    helper performs a single HEAD against the blob; on mismatch it
    re-opens ``path`` and hands the file handle to
    :meth:`BlobStorageResource.upload`, which the Azure SDK streams in
    chunks.  Peak memory through the upload stays at one chunk regardless
    of member size.

    Args:
        blob: Blob storage resource.
        sensitivity: ``"public"`` / ``"confidential"`` / ``"phi"`` (passed
            through to the resource's container resolver).
        prefix: Blob path prefix (already rendered from
            ``contract.prefix_template``).
        filename: Member filename relative to ``prefix``.
        path: Tempfile owned by the extractor generator; valid only
            within the iteration step that yielded it.
        sha256: Hex sha256 computed during extraction.
        size_bytes: Member size in bytes.

    Returns:
        :class:`IngestResult` with ``action="skipped"`` when the existing
        blob's metadata matches, otherwise ``action="uploaded"``.
    """
    blob_path = f"{prefix}/{filename}"
    existing_sha = blob.read_sha256_metadata(sensitivity, blob_path)
    if existing_sha == sha256:
        return IngestResult(path=blob_path, sha256=sha256, action="skipped", size_bytes=size_bytes)
    with path.open("rb") as fp:
        blob.upload(sensitivity, blob_path, fp, sha256=sha256)
    return IngestResult(path=blob_path, sha256=sha256, action="uploaded", size_bytes=size_bytes)
