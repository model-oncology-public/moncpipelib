"""Consumer-side helper: open a partition's blob and its manifest in one call.

Bronze pipelines fed by an ``api_resolver`` ingest contract repeatedly
re-implemented the same five-step boilerplate to get from a
``context.partition_key`` to "bytes I can parse + the manifest fields I
need":

1. ``refs = resolve_source_for_partition(source, partition_key, corpus, blob)``
2. Assert the ref is a :class:`BlobRef` (not a :class:`RawUrl`) and unpack.
3. Open the blob for reading.
4. Locate and parse the per-partition manifest.
5. Pull required keys out of ``manifest.fields`` and raise a descriptive
   error if any are missing.

:func:`read_partition_with_manifest` collapses that into a single
context manager so the call site stays at the level the asset author
actually cares about (a stream + the manifest fields they require).

I/O at boundaries (CLAUDE.md "streaming by default"):

- The manifest is read via :meth:`BlobStorageResource.stream` +
  :meth:`IngestManifest.read_from` -- the same path the resolver uses.
  This preserves the ``manifest_version > KNOWN_MAX_VERSION``
  forward-compat check end-to-end.
- The data blob is exposed as a forward-only ``IO[bytes]`` that the
  caller consumes inside the ``with`` body; peak memory is bounded by
  the blob library's chunk size regardless of file size.

The helper is read-only and consumer-side -- it lives next to
:func:`resolve_source_for_partition`, not next to the writer-side
dispatcher.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import IO, TYPE_CHECKING, Any

from moncpipelib.ingest.exceptions import IngestResolutionError
from moncpipelib.ingest.manifest import IngestManifest
from moncpipelib.ingest.prefix import render_prefix
from moncpipelib.ingest.resolver import resolve_source_for_partition
from moncpipelib.ingest.types import BlobRef

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from moncpipelib.contracts.models import ContractCorpus, DataSource
    from moncpipelib.resources.blob import BlobStorageResource


_MANIFEST_FILENAME: str = "_manifest.json"
"""Mirror of the dispatcher's reserved manifest filename.

Kept here for the same reason :mod:`moncpipelib.ingest.resolver` keeps
its own copy: importing the dispatcher would pull the pattern registry
into read-only consumers that have no business with it.
"""


class ManifestFieldError(IngestResolutionError):
    """A required manifest field is absent or empty.

    Subclasses :class:`IngestResolutionError` so call sites with
    ``except IngestResolutionError`` keep working; callers that want a
    more specific catch handle can use this exception class directly.
    """


@contextmanager
def read_partition_with_manifest(
    *,
    source: DataSource,
    partition_key: str,
    corpus: ContractCorpus,
    blob: BlobStorageResource,
    required_fields: Iterable[str] = (),
    log: Any = None,
) -> Iterator[tuple[BlobRef, IO[bytes], dict[str, Any]]]:
    """Open a partition's blob and its manifest fields in one call.

    Yields ``(ref, blob_stream, manifest_fields)``:

    - ``ref`` -- the resolved :class:`BlobRef` (always one element; the
      resolver enforces exactly-one-match on the underlying glob).
    - ``blob_stream`` -- a forward-only ``IO[bytes]`` over the blob's
      contents.  Peak memory is bounded by the blob library's chunk
      size; the stream is closed when the ``with`` body exits, on
      success or via exception.
    - ``manifest_fields`` -- the manifest's ``fields`` dict (resolver
      output: ``{"release_date": "2026-04-26", ...}``).  Consumers do
      their own date parsing / type coercion.

    Args:
        source: The downstream :class:`DataSource`.
        partition_key: The consumer partition being read.
        corpus: Loaded ingest + source contracts.
        blob: Blob resource used to list, read the manifest, and stream
            the data file.
        required_fields: Manifest field names the caller depends on.  An
            empty iterable (default) skips per-key validation; the
            manifest itself is still required to exist and parse.  A
            field is considered missing if it is absent, ``None``, or
            ``""``; other falsy values (``0``, ``False``, ``[]``) pass.
        log: Optional logger-like object with ``.info(...)``.  When
            supplied, the helper logs ``"Reading {path}
            (sensitivity={sensitivity})"`` at info level.  Matches the
            structural ``LoggingContext`` shape used elsewhere in the
            ingest module -- typed ``Any`` here because callers pass
            ``context.log`` directly (a Dagster logger) rather than a
            wrapper.

    Raises:
        TypeError: When the resolver returns a non-:class:`BlobRef`
            (e.g. a :class:`RawUrl` for a legacy source with
            ``ingest_source is None``).
        ManifestFieldError: When a name in ``required_fields`` is
            absent from ``manifest.fields`` or maps to ``None``/``""``.
            Subclasses :class:`IngestResolutionError`.
        IngestResolutionError: Forwarded from the resolver / manifest
            reader: prefix glob does not match exactly one file, the
            manifest blob is missing (partial-write), the manifest is
            malformed, or ``manifest_version > KNOWN_MAX_VERSION``.
    """
    refs = resolve_source_for_partition(source, partition_key, corpus, blob)
    [ref] = refs  # Resolver always returns exactly one element.
    if not isinstance(ref, BlobRef):
        raise TypeError(
            f"read_partition_with_manifest expects a BlobRef; got "
            f"{type(ref).__name__} for source {source.source_name!r} "
            f"partition {partition_key!r}.  Legacy URL sources "
            f"(ingest_source is None) must use the direct-fetch path."
        )

    # Derive the manifest prefix from the ingest contract -- NOT from
    # ``ref.path.rsplit("/", 1)[0]``.  ``fnmatch``'s ``*`` matches across
    # ``/``, so a glob like ``subdir/*.csv`` resolves to
    # ``{prefix}/subdir/foo.csv`` and the rsplit'd manifest path would
    # become ``{prefix}/subdir/_manifest.json`` instead of the correct
    # ``{prefix}/_manifest.json``.  ``source.ingest_source`` is guaranteed
    # to be set: the resolver would have returned a ``RawUrl`` (caught by
    # the isinstance check above) if it were not.
    assert source.ingest_source is not None
    ingest = corpus.get_ingest(source.ingest_source)
    prefix = render_prefix(ingest.prefix_template, partition_key, ingest)
    manifest_path = f"{prefix}/{_MANIFEST_FILENAME}"

    # Mirror the resolver's manifest read path so the
    # ``manifest_version > KNOWN_MAX_VERSION`` check and the streaming
    # parser stay in effect.  Do NOT ``blob.download(...) +
    # json.loads(...)`` here.
    with blob.stream(ingest.sensitivity, manifest_path) as manifest_fp:
        manifest = IngestManifest.read_from(manifest_fp)

    _validate_required_fields(manifest.fields, required_fields, manifest_path)

    if log is not None:
        log.info(
            "Reading %s (sensitivity=%s)",
            ref.path,
            ref.sensitivity,
        )

    with blob.stream(ref.sensitivity, ref.path) as data_fp:
        yield ref, data_fp, manifest.fields


@contextmanager
def download_partition_parts_with_manifest(
    *,
    source: DataSource,
    partition_key: str,
    corpus: ContractCorpus,
    blob: BlobStorageResource,
    required_fields: Iterable[str] = (),
    log: Any = None,
) -> Iterator[tuple[list[Path], dict[str, Any]]]:
    """Download a partition's N part files to local tempfiles + its manifest fields.

    The multi-file counterpart to :func:`read_partition_with_manifest`
    (#438 + #439).  Where that helper streams a single blob forward-only,
    this one resolves the partition's part set (``match: many`` sources
    return N :class:`BlobRef`\\ s) and downloads each to a **seekable
    local file** -- required because parquet's footer lives at
    end-of-file, so a forward-only blob stream cannot be scanned.  The
    yielded paths feed directly into
    :func:`~moncpipelib.streaming.stream_parquet_batches`.

    Yields ``(paths, manifest_fields)``:

    - ``paths`` -- local tempfile paths, in resolver order (sorted by
      blob path, so ``part-00001`` ... stream deterministically).  For a
      ``match: one`` source this is a one-element list.  The files are
      deleted when the ``with`` body exits, on success or via exception,
      so consumers must scan them inside the block.
    - ``manifest_fields`` -- the manifest's ``fields`` dict.

    Peak memory is bounded by the blob chunk size per file (each part
    streams to disk via :meth:`BlobStorageResource.download_to_path`); the
    parquet scan over the returned paths is separately row-bounded.

    Raises:
        TypeError: When the resolver returns a non-:class:`BlobRef` (a
            legacy :class:`RawUrl` source).
        ManifestFieldError / IngestResolutionError: As in
            :func:`read_partition_with_manifest` (missing required field,
            wrong match cardinality, missing/malformed manifest).
    """
    refs = resolve_source_for_partition(source, partition_key, corpus, blob)
    blob_refs: list[BlobRef] = []
    for ref in refs:
        if not isinstance(ref, BlobRef):
            raise TypeError(
                f"download_partition_parts_with_manifest expects BlobRefs; got "
                f"{type(ref).__name__} for source {source.source_name!r} "
                f"partition {partition_key!r}.  Legacy URL sources "
                f"(ingest_source is None) are not supported."
            )
        blob_refs.append(ref)

    assert source.ingest_source is not None  # guaranteed: refs are BlobRefs
    ingest = corpus.get_ingest(source.ingest_source)
    prefix = render_prefix(ingest.prefix_template, partition_key, ingest)
    manifest_path = f"{prefix}/{_MANIFEST_FILENAME}"
    with blob.stream(ingest.sensitivity, manifest_path) as manifest_fp:
        manifest = IngestManifest.read_from(manifest_fp)

    _validate_required_fields(manifest.fields, required_fields, manifest_path)

    if log is not None:
        log.info(
            "Downloading %d part(s) for partition %s (sensitivity=%s)",
            len(blob_refs),
            partition_key,
            ingest.sensitivity,
        )

    with tempfile.TemporaryDirectory(prefix="moncpipelib-parts-") as tmpdir:
        paths: list[Path] = []
        for index, ref in enumerate(blob_refs):
            # Index-prefix the local name so two source parts with the
            # same basename never collide on disk, while preserving the
            # original basename for debuggability.
            dest = Path(tmpdir) / f"{index:05d}-{PurePosixPath(ref.path).name}"
            blob.download_to_path(ref.sensitivity, ref.path, dest)
            paths.append(dest)
        yield paths, manifest.fields


def _validate_required_fields(
    fields: dict[str, Any],
    required: Iterable[str],
    manifest_path: str,
) -> None:
    """Raise :class:`ManifestFieldError` for the first absent or empty field.

    "Absent" means the key is missing, ``None``, or ``""``.  Other falsy
    values (``0``, ``False``, ``[]``) pass -- contracts realistically
    carry numeric or boolean fields and a blanket ``if not value:``
    would reject them.  Explicit per-key validation, in input order.
    """
    for name in required:
        if name not in fields:
            raise ManifestFieldError(
                f"Manifest at {manifest_path!r} is missing required field "
                f"{name!r}; available fields: {sorted(fields)}"
            )
        value = fields[name]
        if value is None or value == "":
            raise ManifestFieldError(
                f"Manifest at {manifest_path!r} has empty value for required "
                f"field {name!r} (value={value!r}); available fields: "
                f"{sorted(fields)}"
            )
