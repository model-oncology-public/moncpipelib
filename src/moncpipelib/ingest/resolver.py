"""Consumer-side blob resolution for the ingest boundary.

Downstream pipelines call :func:`resolve_source_for_partition` as their
only entry point into the ingest boundary. They never see the ingest
pattern (``http_urls``, ``api_resolver``, ...) that landed the data --
their code shape is uniform across all pattern-fed sources.

Phase 1 covered the static case (``http_urls`` + enumerated periods);
Phase 2 (this module's update) reads the per-partition manifest written
by :func:`~moncpipelib.ingest.dispatcher.materialize_with_manifest` to
hydrate the ``FromIngestTemplate`` consumer branch -- previously
:class:`NotImplementedError`.
"""

from __future__ import annotations

from fnmatch import fnmatch
from typing import TYPE_CHECKING

from moncpipelib.ingest.exceptions import IngestResolutionError
from moncpipelib.ingest.manifest import IngestManifest
from moncpipelib.ingest.prefix import render_prefix
from moncpipelib.ingest.types import BlobRef, RawUrl

if TYPE_CHECKING:
    from moncpipelib.contracts.models import (
        ContractCorpus,
        DataSource,
        FromIngestTemplate,
        IngestContract,
        Period,
    )
    from moncpipelib.resources.blob import BlobStorageResource


_MANIFEST_FILENAME: str = "_manifest.json"
"""Mirror of the dispatcher's reserved manifest filename.  Kept here
to avoid an import-time dependency from this module on
:mod:`moncpipelib.ingest.dispatcher` (the dispatcher imports the
patterns registry, which we want to keep optional for read-only
consumers)."""


def resolve_source_for_partition(
    source: DataSource,
    partition_key: str,
    corpus: ContractCorpus,
    blob: BlobStorageResource,
) -> list[BlobRef | RawUrl]:
    """Resolve blob refs (or legacy URLs) for a consumer partition.

    Args:
        source: The downstream :class:`DataSource` to resolve.
        partition_key: The consumer partition being materialized.
        corpus: Loaded ingest + source contracts.  Used to look up the
            linked ingest when ``source.ingest_source`` is set.
        blob: Blob resource used to list the ingest prefix and (for
            the ``FromIngestTemplate`` branch) read the per-partition
            manifest.

    Returns:
        A list of refs:

        * ``[RawUrl]`` (one element) when the source has no ingest link
          (legacy path).
        * ``[BlobRef]`` (one element) when the source is wired through
          the ingest boundary and the period's / template's glob is
          ``match="one"`` (default) -- exactly one file must match.
        * ``[BlobRef, ...]`` (N elements, sorted by path) when the
          period / template declares ``match="many"`` (#438) -- N part
          files per partition, >=1 required.

    Raises:
        IngestResolutionError: When the ingest prefix yields the wrong
            number of files for the period's glob (drift detection: !=1
            for ``match="one"``, 0 for ``match="many"``),
            when the partition's manifest is missing
            (``FromIngestTemplate`` branch -- partial-write detected),
            when the manifest is malformed or its
            ``manifest_version`` is too new, or when the
            ``effective_from_field`` is missing from manifest fields.
    """
    from moncpipelib.contracts.models import FromIngestTemplate

    if source.ingest_source is None:
        period = _find_period(source, partition_key)
        return [RawUrl(period.source)]

    ingest = corpus.get_ingest(source.ingest_source)
    prefix = render_prefix(ingest.prefix_template, partition_key, ingest)

    if isinstance(source.periods, FromIngestTemplate):
        return list(_resolve_from_template(source.periods, ingest, prefix, blob))

    period = _find_period(source, partition_key)
    glob = period.source
    return list(_resolve_static_glob(ingest, prefix, glob, blob, period.match))


def _resolve_from_template(
    template: FromIngestTemplate,
    ingest: IngestContract,
    prefix: str,
    blob: BlobStorageResource,
) -> list[BlobRef]:
    """Resolve a ``FromIngestTemplate`` source via the per-partition manifest.

    1. Load ``{prefix}/_manifest.json`` (raises :class:`IngestResolutionError`
       if absent -- signals partial-write recovery state).
    2. Verify ``effective_from_field`` is populated in
       ``manifest.fields`` (drift detection: a renamed resolver field
       breaks downstream consumers, so fail at resolve time rather than
       silently returning ``None``).
    3. Render ``template.source`` with ``manifest.fields`` substitutions
       (the glob may reference ``{release_version}`` etc.).
    4. Match landed blobs against the rendered glob and enforce
       exactly-one (same drift contract as the static branch).
    """
    manifest = _load_manifest(blob, ingest, prefix)

    if template.effective_from_field not in manifest.fields:
        raise IngestResolutionError(
            f"Manifest at prefix {prefix!r} is missing "
            f"effective_from_field={template.effective_from_field!r}; "
            f"available fields: {sorted(manifest.fields)}"
        )

    try:
        rendered_glob = template.source.format_map(manifest.fields)
    except KeyError as e:
        raise IngestResolutionError(
            f"FromIngestTemplate.source {template.source!r} references "
            f"manifest field {e} that is not present; available fields: "
            f"{sorted(manifest.fields)}"
        ) from e

    return _resolve_static_glob(ingest, prefix, rendered_glob, blob, template.match)


def _load_manifest(
    blob: BlobStorageResource,
    ingest: IngestContract,
    prefix: str,
) -> IngestManifest:
    """Load and parse the manifest for a partition prefix.

    Raises :class:`IngestResolutionError` when:

    - The manifest blob is absent (partial-write detected).
    - The JSON is malformed, a required field is missing, or the
      ``manifest_version`` exceeds :data:`KNOWN_MAX_VERSION`
      (forwarded by :meth:`IngestManifest.read_from`).
    """
    manifest_path = f"{prefix}/{_MANIFEST_FILENAME}"
    if not blob.exists(ingest.sensitivity, manifest_path):
        raise IngestResolutionError(
            f"Manifest not found at {manifest_path!r}; partition has files "
            f"but no manifest -- partial-write detected.  Re-run the "
            f"materialization to land the manifest."
        )
    # Stream-parse the manifest off the wire (#243 / Migration 012
    # Phase B): the files array is yielded entry-by-entry rather than
    # materializing the full list in memory before construction.  A
    # 100k+ file partition keeps peak heap bounded by the streaming
    # chunk size.
    with blob.stream(ingest.sensitivity, manifest_path) as fp:
        return IngestManifest.read_from(fp)


def _resolve_static_glob(
    ingest: IngestContract,
    prefix: str,
    glob: str,
    blob: BlobStorageResource,
    match: str = "one",
) -> list[BlobRef]:
    """Match ``glob`` under ``prefix`` and enforce the ``match`` cardinality.

    ``match="one"`` (default) requires exactly one match -- 0 = not yet
    materialized / partial-write, >1 = upstream drift or a sibling file --
    and returns a one-element list.  ``match="many"`` (#438) permits N
    matches (>=1) for asset folders delivered as N part files, returning
    the matches sorted by path so a multi-part scan is deterministic; 0
    still raises (nothing landed).

    Excludes the manifest filename so a permissive glob like ``*`` does
    not collide with the manifest blob.
    """
    # Iterate lazily via ``iter_list`` so a high-cardinality partition
    # prefix (UMLS Metathesaurus approaches 100k+ files once unpacked)
    # does not materialize every blob name before fnmatch filtering.
    # Migration 012 Phase E (#246).
    matches = [
        path
        for path in blob.iter_list(ingest.sensitivity, prefix)
        if fnmatch(path, f"{prefix}/{glob}") and not path.endswith(f"/{_MANIFEST_FILENAME}")
    ]
    if match == "many":
        if not matches:
            raise IngestResolutionError(
                f"Expected >=1 match for glob {glob!r} under prefix {prefix!r} "
                f"(sensitivity={ingest.sensitivity}, match=many); got 0"
            )
        return [BlobRef(sensitivity=ingest.sensitivity, path=path) for path in sorted(matches)]
    if len(matches) != 1:
        raise IngestResolutionError(
            f"Expected exactly 1 match for glob {glob!r} under prefix {prefix!r} "
            f"(sensitivity={ingest.sensitivity}); got {len(matches)}: {matches}"
        )
    return [BlobRef(sensitivity=ingest.sensitivity, path=matches[0])]


def _find_period(source: DataSource, partition_key: str) -> Period:
    """Locate a ``Period`` by partition key within a list-typed source.

    Raises ``IngestResolutionError`` if the source carries a
    ``FromIngestTemplate`` instead of a list, or if the key is absent.
    """
    from moncpipelib.contracts.models import FromIngestTemplate

    if isinstance(source.periods, FromIngestTemplate):
        raise IngestResolutionError(
            f"DataSource {source.source_name!r} uses FromIngestTemplate; "
            f"cannot look up partition {partition_key!r} without the ingest manifest"
        )
    for period in source.periods:
        if period.partition_key == partition_key:
            return period
    raise IngestResolutionError(
        f"DataSource {source.source_name!r} has no period with partition_key={partition_key!r}"
    )
