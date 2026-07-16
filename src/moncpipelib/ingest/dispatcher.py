"""Materialize-with-manifest dispatcher for the universal ingest boundary.

This module provides :func:`materialize_with_manifest`, the single
entry point ingest assets call.  It wraps any
:class:`~moncpipelib.ingest.patterns.IngestPattern` and centralizes the
manifest write so individual pattern implementations cannot forget it.

Atomicity contract (per #216):

- The pattern's ``materialize_partition`` runs first.  If it raises,
  the dispatcher does NOT write the manifest -- the partition is left
  in the "files but no manifest" intermediate state.  This is the
  documented partial-write recovery mode: a consumer that calls
  :func:`~moncpipelib.ingest.resolver.resolve_source_for_partition`
  on a partition with files but no manifest receives
  :class:`~moncpipelib.ingest.exceptions.IngestResolutionError`.
- Re-running the dispatcher on a partial partition skips already-landed
  files via the pattern's ``hash_compare`` idempotency, uploads any
  missing files, and then writes the manifest -- closing the window.
- The manifest's own bytes are sha256'd before upload so the manifest
  blob carries integrity metadata too (HIPAA 164.312(c)(1) per-object
  control).

Compliance:

- The dispatcher does not log secret values.  ``ctx.run_id`` is
  surfaced in audit logs for run correlation.
- Manifest payload is bounded by the partition's file count; a 5+ GB
  UMLS partition produces a ~MB manifest, easily JSON-serialized in
  memory.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

from moncpipelib.ingest._hashing import hashing_tempfile
from moncpipelib.ingest.manifest import IngestManifest, ManifestFileEntry
from moncpipelib.ingest.prefix import render_prefix


def _coerce_jsonable(value: Any) -> Any:
    """Recursively normalize ``resolver_config`` to JSON-friendly scalars.

    PyYAML parses bare ISO dates (``2024-01-01``) as :class:`datetime.date`
    and full ISO timestamps as :class:`datetime.datetime`.  Contract
    authors may also reach for :class:`uuid.UUID` or :class:`decimal.Decimal`.
    None of those round-trip through ``json.dumps`` without help.

    Coercing here -- before constructing :class:`IngestManifest` -- keeps
    the in-memory dataclass and the round-tripped manifest equal under
    ``==``.  :func:`_json_default` in ``manifest.py`` is the safety net
    for any direct ``write_to`` caller; this is the primary normalizer.

    See issue #233 for the bug that motivated this.
    """
    if isinstance(value, dict):
        return {k: _coerce_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_jsonable(v) for v in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Decimal):
        return str(value)
    return value


if TYPE_CHECKING:
    from pathlib import Path

    from moncpipelib.contracts.models import IngestContract
    from moncpipelib.ingest.patterns import IngestPattern
    from moncpipelib.ingest.types import (
        IngestContext,
        IngestResult,
        PartitionSpec,
    )
    from moncpipelib.resources.blob import BlobStorageResource


_MANIFEST_FILENAME: str = "_manifest.json"
"""Reserved filename for the per-partition manifest.

Lives at ``{prefix}/_manifest.json``.  Underscore prefix follows
existing conventions (consumer globs typically don't match
``_*.json``).  Reserved name -- contracts must not place period
``source`` globs that would intentionally match this file.
"""


def materialize_with_manifest(
    pattern: IngestPattern,
    contract: IngestContract,
    partition_spec: PartitionSpec,
    blob: BlobStorageResource,
    ctx: IngestContext,
) -> list[IngestResult]:
    """Materialize a partition and atomically write its manifest.

    Calling sequence:

    1. ``pattern.materialize_partition(contract, partition_spec, blob, ctx)``
       runs first.  Any exception propagates UNCAUGHT -- the manifest
       is NOT written, leaving the partition in the partial-write
       intermediate state.
    2. On success, the dispatcher builds an :class:`IngestManifest`
       from the returned :class:`IngestResult` list plus
       ``partition_spec.metadata`` and writes it to
       ``{prefix}/_manifest.json``.
    3. Returns the original results list unchanged.

    This is the single entry point ingest assets should call.  Pattern
    ``materialize_partition`` is no longer invoked directly by
    data-platform code.

    Args:
        pattern: The :class:`IngestPattern` implementation
            (typically obtained via
            :func:`~moncpipelib.ingest.patterns.get_pattern`).
        contract: The :class:`~moncpipelib.contracts.models.IngestContract`.
        partition_spec: The :class:`PartitionSpec` for the partition
            being materialized.
        blob: The :class:`BlobStorageResource` for the landing
            container.
        ctx: The :class:`IngestContext` (log + secrets + run_id).

    Returns:
        The list of :class:`IngestResult` objects from the pattern
        (one per landed file; ``"uploaded"`` or ``"skipped"`` per
        the pattern's hash_compare contract).
    """
    results = pattern.materialize_partition(contract, partition_spec, blob, ctx)

    manifest = _build_manifest(pattern, contract, partition_spec, results, ctx)
    prefix = render_prefix(contract.prefix_template, partition_spec.key, contract)
    manifest_path = f"{prefix}/{_MANIFEST_FILENAME}"

    # Stream the manifest to a tempfile while hashing in the same pass
    # (#243 / Migration 012 Phase B).  The manifest never exists as a
    # full ``bytes`` object on the heap, so a 100k+ entry partition
    # manifest stays bounded by the streaming chunk size rather than
    # the manifest's serialized size.
    with (
        _write_manifest_to_tempfile(manifest) as (manifest_local_path, sha256),
        manifest_local_path.open("rb") as fp,
    ):
        blob.upload(
            contract.sensitivity,
            manifest_path,
            fp,
            sha256=sha256,
        )

    return results


@contextmanager
def _write_manifest_to_tempfile(
    manifest: IngestManifest,
) -> Iterator[tuple[Path, str]]:
    """Stream a manifest to a tempfile while computing sha256 in one pass.

    Yields ``(path, sha256_hex)``.  The tempfile is unlinked on context
    exit so a failure mid-stream does not leak partial bytes onto the
    pod's ephemeral disk.

    Thin wrapper over :func:`~moncpipelib.ingest._hashing.hashing_tempfile`
    (the hash-while-writing idiom shared with #239's extracted archive
    members and #415's crawl assembly); the manifest write path keeps
    the same memory profile (peak heap bounded by chunk size, not
    manifest size).
    """
    with hashing_tempfile(suffix=".manifest.json") as writer:
        # ``IngestManifest.write_to`` calls only ``write(bytes) -> int``,
        # which the writer satisfies without the full ``IO[bytes]`` shape.
        manifest.write_to(writer)  # type: ignore[arg-type]
        writer.close()
        yield writer.path, writer.sha256_hexdigest()


def _build_manifest(
    pattern: IngestPattern,
    contract: IngestContract,
    partition_spec: PartitionSpec,
    results: list[IngestResult],
    ctx: IngestContext,
) -> IngestManifest:
    """Build the :class:`IngestManifest` for a freshly-materialized partition.

    The ``resolver`` block is sourced from the pattern's optional
    ``manifest_resolver_block(contract)`` method (per #415 -- same
    ``getattr`` opt-in style as ``partition_metadata``), so each pattern
    names whichever extension point produced the partition:
    ``api_resolver`` returns its resolver name + config, ``api_crawl``
    its crawl plan.  Patterns without the method get the generic
    ``{"name": <pattern.name>, "config": {}}`` fallback (``http_urls``)
    -- audit symmetry across patterns.  Same redaction contract as
    resolver output: blocks are persisted durably in ``_manifest.json``
    and MUST NOT contain api_keys, signed URLs, or PHI.

    The ``fields`` block carries the partition's release dict so
    consumers using
    :class:`~moncpipelib.contracts.models.FromIngestTemplate` can
    hydrate ``effective_from`` from manifest fields.

    Per #256, ``fields`` is sourced via the pattern's optional
    ``partition_metadata(contract, partition_key, ctx)`` method when
    present (re-resolves the release dict at manifest-write time, since
    Dagster's dynamic-partitions registry only persists the key, not
    the spec metadata from discovery).  The dispatcher falls back to
    ``partition_spec.metadata`` when the method is absent OR returns an
    empty dict -- preserving back-compat for callers that populate spec
    metadata directly (e.g. tests).
    """
    resolver_block: dict[str, Any]
    block_fn = getattr(pattern, "manifest_resolver_block", None)
    if callable(block_fn):
        raw_block = dict(block_fn(contract))
        # Normalize shape at the dispatcher so the durable manifest
        # schema (name: str, config: mapping) holds regardless of
        # pattern authorship; config values coerced per #233.
        resolver_block = {
            "name": str(raw_block.get("name", pattern.name)),
            "config": _coerce_jsonable(dict(raw_block.get("config") or {})),
        }
    else:
        resolver_block = {"name": pattern.name, "config": {}}

    files = tuple(
        ManifestFileEntry(path=r.path, sha256=r.sha256, size_bytes=r.size_bytes)
        for r in results
        # Defensive: if a pattern ever returns a manifest entry, skip it
        # so the manifest never lists itself.
        if not r.path.endswith(f"/{_MANIFEST_FILENAME}")
    )

    # Per #256: pattern method is the primary source of manifest fields.
    # ``getattr`` keeps the IngestPattern Protocol stable (mirrors the
    # ``discovery_requires_auth`` opt-in pattern from #253) so test stubs
    # and any future third-party patterns aren't forced to implement it.
    fields_dict: dict[str, Any] = {}
    metadata_fn = getattr(pattern, "partition_metadata", None)
    if callable(metadata_fn):
        fields_dict = dict(metadata_fn(contract, partition_spec.key, ctx))
    if not fields_dict:
        # Back-compat fallback: any caller that populated
        # ``partition_spec.metadata`` directly (e.g. unit tests, or a
        # future ad-hoc dispatcher caller) keeps working.
        fields_dict = dict(partition_spec.metadata or {})

    return IngestManifest(
        manifest_version=1,
        source_id=contract.source_id,
        source_name=contract.source_name,
        partition_key=partition_spec.key,
        materialized_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        resolver=resolver_block,
        # Coerce field values symmetrically with ``resolver_config``
        # (#233): a pattern / resolver that returns a date / datetime /
        # UUID / Decimal would otherwise have ``_json_default`` rescue
        # the on-disk write but leave the in-memory dataclass NOT
        # round-tripping equal -- the same authoring trap the
        # resolver_config coercion was meant to eliminate.
        fields=_coerce_jsonable(fields_dict),
        files=files,
    )
