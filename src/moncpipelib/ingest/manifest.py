"""Per-partition manifest for the universal ingest boundary.

Every partition materialized through ``materialize_with_manifest`` writes
exactly one ``_manifest.json`` document at the partition prefix.  The
manifest is the canonical "this partition is fully materialized" marker:

- Consumers calling :func:`~moncpipelib.ingest.resolver.resolve_source_for_partition`
  on a partition with a :class:`~moncpipelib.contracts.models.FromIngestTemplate`
  source read the manifest to hydrate the template's
  ``effective_from_field`` and to enumerate landed files.
- A partition with files but no manifest is an intermediate state
  (a prior materialization was interrupted); consumers detect this and
  raise :class:`~moncpipelib.ingest.exceptions.IngestResolutionError`.
- The reader version-checks ``manifest_version`` on read so future
  schema changes fail loud rather than silent.

I/O at boundaries (#243 / Migration 012 Phase B):

- :meth:`IngestManifest.write_to` streams the JSON document directly to
  an ``IO[bytes]`` writer.  The ``files`` array is emitted entry-by-entry
  so memory stays bounded by one entry's serialized size, regardless of
  how many entries the manifest carries.  Output is byte-identical to
  the canonical ``json.dumps(asdict(self), sort_keys=True, indent=2)``
  -- the manifest blob's sha256 is part of the on-disk contract and
  must remain stable across implementation changes.
- :meth:`IngestManifest.read_from` parses the document via ``ijson``,
  yielding ``files`` entries lazily so a 100k-file manifest never
  materializes the full array on the heap.

Compliance:

- Atomic-write semantics (manifest written only after every file lands)
  give per-partition integrity (HIPAA 164.312(c)(1)) on top of the
  per-object sha256 already stored in blob metadata.
- Forward-compat is explicit: ``manifest_version > KNOWN_MAX_VERSION``
  is a read-time failure with a clear error, not a silent behavior
  change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import IO, Any
from uuid import UUID

import ijson

from moncpipelib.ingest.exceptions import IngestResolutionError


def _json_default(obj: Any) -> str:
    """``json.dumps`` ``default=`` hook for manifest serialization.

    Handles types PyYAML or contract authors realistically place in
    ``resolver_config`` (bare ISO dates, ISO timestamps, UUIDs, decimals)
    that ``json.dumps`` does not encode out of the box.  ``datetime`` is
    a subclass of ``date``; ``isoformat`` works for both.

    Anything else falls through to a ``TypeError`` -- the dispatcher's
    ``_coerce_jsonable`` is the primary normalizer; this default exists
    so direct callers of :meth:`IngestManifest.write_to` (tests,
    debugging) also get a useful encoding rather than a crash, while
    still surfacing genuinely unsupported types.
    """
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


KNOWN_MAX_VERSION: int = 1
"""Highest ``manifest_version`` this build can read.

Bump when adding fields to :class:`IngestManifest`.  The reader rejects
manifests with ``manifest_version > KNOWN_MAX_VERSION`` -- forward-compat
must be explicit (see module docstring).
"""


@dataclass(frozen=True, slots=True)
class ManifestFileEntry:
    """One landed file inside an ingest partition manifest."""

    path: str
    """Blob path relative to the container."""

    sha256: str
    """Hex-encoded SHA-256 of the file's content.

    Matches the blob's ``x-ms-meta-sha256`` metadata so a consumer can
    cross-check object integrity against the manifest without an extra
    download.
    """

    size_bytes: int
    """File size in bytes."""


@dataclass(frozen=True, slots=True)
class IngestManifest:
    """Canonical per-partition manifest written by ``materialize_with_manifest``.

    Written only after every file in the partition lands successfully.
    Read by :func:`~moncpipelib.ingest.resolver.resolve_source_for_partition`
    to hydrate the :class:`~moncpipelib.contracts.models.FromIngestTemplate`
    consumer branch.

    Attributes:
        manifest_version: Schema version of THIS manifest.  Readers MUST
            reject ``manifest_version > KNOWN_MAX_VERSION``.  Older
            versions are tolerated only when the schema is strictly
            additive.
        source_id: Stable UUID of the ingest source (matches
            ``IngestContract.source_id``).
        source_name: Human-readable name (matches
            ``IngestContract.source_name``).
        partition_key: The partition this manifest describes.
        materialized_at: ISO-8601 UTC timestamp
            (``"YYYY-MM-DDTHH:MM:SSZ"``).
        resolver: Resolver name + config when materialized via
            ``api_resolver``; ``{"name": "http_urls", "config": {}}``
            when materialized via the static-URL pattern.  Audit trail.
        fields: Resolver-derived dynamic fields (e.g.
            ``{"release_date": "2026-04-26", "release_version": "2026AB"}``).
            Consumed by :class:`~moncpipelib.contracts.models.FromIngestTemplate`
            via its ``effective_from_field``.
        files: Tuple of one entry per landed file.  The authoritative
            list for partition-completeness checks.
    """

    manifest_version: int
    source_id: str
    source_name: str
    partition_key: str
    materialized_at: str
    resolver: dict[str, Any]
    fields: dict[str, Any]
    files: tuple[ManifestFileEntry, ...]

    def write_to(self, fp: IO[bytes]) -> None:
        """Stream the manifest as JSON to ``fp``.

        Output is byte-identical to ``json.dumps(asdict(self),
        sort_keys=True, indent=2, default=_json_default)`` for any
        manifest -- a fixture-based byte-stability test in
        ``tests/test_manifest_streaming.py`` pins this invariant so the
        manifest blob's sha256 stays stable across implementation
        changes.

        The ``files`` array is emitted entry-by-entry so peak memory
        tracks one entry's serialized size (a few hundred bytes), not
        the full manifest's size.  All other top-level fields
        (``resolver``, ``fields``, scalars) are bounded by configuration
        and serialized whole.
        """
        # Top-level keys EXCEPT files.  These are bounded by
        # configuration and serialized whole.
        non_files: dict[str, Any] = {
            "manifest_version": self.manifest_version,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "partition_key": self.partition_key,
            "materialized_at": self.materialized_at,
            "resolver": dict(self.resolver),
            "fields": dict(self.fields),
        }
        sorted_keys = sorted([*non_files.keys(), "files"])

        fp.write(b"{\n")
        last_idx = len(sorted_keys) - 1
        for i, key in enumerate(sorted_keys):
            if key == "files":
                _write_files_streaming(fp, self.files)
            else:
                _write_kv_at_indent_2(fp, key, non_files[key])
            fp.write(b",\n" if i < last_idx else b"\n")
        fp.write(b"}")

    @classmethod
    def read_from(cls, fp: IO[bytes]) -> IngestManifest:
        """Stream-parse a manifest from ``fp``.

        Files are yielded lazily through ``ijson``; resolver/fields are
        bounded by configuration and built whole.  The reader's error
        surface: malformed JSON, missing required fields, malformed
        file entries, and ``manifest_version > KNOWN_MAX_VERSION`` all
        raise :class:`IngestResolutionError`.

        Raises:
            IngestResolutionError: When the JSON is malformed, a
                required field is missing, the file list contains a
                malformed entry, the root is not a JSON object, or
                ``manifest_version > KNOWN_MAX_VERSION``.
        """
        files: list[ManifestFileEntry] = []
        # `state` holds the top-level dict (minus the streamed files).
        # The sentinel distinguishes "didn't see a top-level object"
        # (e.g. root was an array, scalar, or empty stream) from "saw
        # an object that happens to be empty".
        _UNSET = object()
        state: Any = _UNSET

        # Builder stack: each entry is either a dict, list, or the
        # _FILES_SENTINEL.  When we close a container, we attach it to
        # its parent (or, at root, set ``state``); ``files.item`` end_map
        # is special-cased to emit a ManifestFileEntry instead.
        _FILES_SENTINEL = object()
        stack: list[Any] = []
        pending_keys: list[str | None] = []
        # Pin the partition-completeness invariant: streaming through
        # the array means ``files`` does NOT land in ``state``, so a
        # missing-files-key manifest would otherwise parse as
        # ``files=()`` -- a HIPAA 164.312(c)(1) integrity gap (a
        # malformed manifest could resolve as completeness-checked when
        # it isn't).  This flag distinguishes "saw the array, it was
        # empty" from "no array at all".
        saw_files_array = False

        def _attach_to_parent(value: Any) -> None:
            if not stack:
                return
            parent = stack[-1]
            if parent is _FILES_SENTINEL:
                return  # files array contents are emitted, not collected
            if isinstance(parent, dict):
                k = pending_keys[-1]
                if k is None:
                    raise IngestResolutionError("Manifest parser saw a value with no preceding key")
                parent[k] = value
                pending_keys[-1] = None
            elif isinstance(parent, list):
                parent.append(value)

        try:
            for prefix, event, value in ijson.parse(fp):
                if event == "start_map":
                    stack.append({})
                    pending_keys.append(None)
                elif event == "start_array":
                    if prefix == "files":
                        stack.append(_FILES_SENTINEL)
                        saw_files_array = True
                    else:
                        stack.append([])
                    pending_keys.append(None)
                elif event == "end_map":
                    completed = stack.pop()
                    pending_keys.pop()
                    if prefix == "files.item":
                        try:
                            files.append(
                                ManifestFileEntry(
                                    path=str(completed["path"]),
                                    sha256=str(completed["sha256"]),
                                    size_bytes=int(completed["size_bytes"]),
                                )
                            )
                        except (KeyError, ValueError) as e:
                            raise IngestResolutionError(
                                f"Manifest 'files' entry malformed: {e}"
                            ) from e
                    elif not stack:
                        # Root closed.
                        state = completed
                    else:
                        _attach_to_parent(completed)
                elif event == "end_array":
                    completed = stack.pop()
                    pending_keys.pop()
                    if completed is not _FILES_SENTINEL:
                        _attach_to_parent(completed)
                elif event == "map_key":
                    pending_keys[-1] = value
                else:
                    # Scalar event: integer, number, string, boolean, null.
                    if stack:
                        _attach_to_parent(value)
        except ijson.JSONError as e:
            raise IngestResolutionError(f"Manifest is not valid JSON: {e}") from e

        if state is _UNSET or not isinstance(state, dict):
            actual = type(state).__name__ if state is not _UNSET else "non-object root"
            raise IngestResolutionError(f"Manifest must be a JSON object; got {actual}")

        # Version check first -- the most structurally diagnostic of
        # the field validations.  Forward-compat ("we don't read v2")
        # should fire even if other fields are also broken; mirrors the
        # legacy validation order so error messages stay stable for
        # consumers that key on them.
        version = state.get("manifest_version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise IngestResolutionError(
                f"Manifest 'manifest_version' must be an integer; got {version!r}"
            )
        if version > KNOWN_MAX_VERSION:
            raise IngestResolutionError(
                f"Manifest 'manifest_version' is {version}; this build supports "
                f"up to {KNOWN_MAX_VERSION}.  Upgrade moncpipelib to read it."
            )

        # ``files`` must be a JSON array.  The streaming path only emits
        # ``ManifestFileEntry`` for ``start_array`` events at prefix
        # ``"files"`` -- a dict or scalar at that key would have been
        # routed through the generic builder and ended up in
        # ``state["files"]`` instead.  The array-shape contract is part
        # of the manifest schema, so reject the malformed shape loudly.
        if "files" in state:
            raise IngestResolutionError(
                f"Manifest 'files' must be a JSON array; got {type(state['files']).__name__}"
            )
        # And reject the entirely-missing case: streaming through the
        # array means `files` never lands in `state`, so the absence
        # check above does not catch "no key at all" -- pin it via
        # ``saw_files_array``.
        if not saw_files_array:
            raise IngestResolutionError("Manifest missing required field: 'files'")

        try:
            resolver = state["resolver"]
            fields = state["fields"]
            if not isinstance(resolver, dict):
                raise TypeError(f"'resolver' must be a mapping; got {type(resolver).__name__}")
            if not isinstance(fields, dict):
                raise TypeError(f"'fields' must be a mapping; got {type(fields).__name__}")
            return cls(
                manifest_version=version,
                source_id=str(state["source_id"]),
                source_name=str(state["source_name"]),
                partition_key=str(state["partition_key"]),
                materialized_at=str(state["materialized_at"]),
                resolver=dict(resolver),
                fields=dict(fields),
                files=tuple(files),
            )
        except KeyError as e:
            raise IngestResolutionError(f"Manifest missing required field: {e}") from e
        except (TypeError, ValueError) as e:
            raise IngestResolutionError(f"Manifest field has wrong type: {e}") from e


def _write_kv_at_indent_2(fp: IO[bytes], key: str, value: Any) -> None:
    """Emit ``  "<key>": <serialized value>`` at indent level 2.

    The serialized value is ``json.dumps(value, sort_keys=True,
    indent=2, default=_json_default)`` re-indented so subsequent lines
    sit at the parent's indent level rather than column 0.  Output
    matches what ``json.dumps`` would emit for the value embedded in a
    parent at indent level 2.
    """
    rendered = json.dumps(value, sort_keys=True, indent=2, default=_json_default)
    lines = rendered.split("\n")
    if len(lines) == 1:
        # Scalar (or empty container like {} or []) -- single-line value.
        fp.write(f'  "{key}": {lines[0]}'.encode())
        return
    # Multi-line value (nested dict or array).  First line stays as-is
    # after the key; subsequent lines get 2 extra spaces so they sit at
    # the parent's indent level.
    first = f'  "{key}": {lines[0]}'
    rest = "\n".join("  " + line for line in lines[1:])
    fp.write((first + "\n" + rest).encode())


def _write_files_streaming(fp: IO[bytes], files: tuple[ManifestFileEntry, ...]) -> None:
    """Emit ``  "files": [<entries>]`` at indent level 2, streaming entries.

    Output matches what ``json.dumps`` would produce for ``"files":
    [<list of dicts>]`` embedded in a parent at indent level 2: the
    array opens on the same line as ``"files":``, each entry sits at
    indent 4, and the closing ``]`` returns to indent 2.
    """
    if not files:
        # ``json.dumps`` collapses empty arrays to ``[]`` on one line.
        fp.write(b'  "files": []')
        return
    fp.write(b'  "files": [\n')
    last_idx = len(files) - 1
    for i, entry in enumerate(files):
        entry_dict = {
            "path": entry.path,
            "sha256": entry.sha256,
            "size_bytes": entry.size_bytes,
        }
        rendered = json.dumps(entry_dict, sort_keys=True, indent=2, default=_json_default)
        # Re-indent every line by 4 extra spaces (array elements sit at
        # indent 4; their contents at indent 6; closing brace at indent 4).
        indented = "\n".join("    " + line for line in rendered.split("\n"))
        fp.write(indented.encode())
        fp.write(b",\n" if i < last_idx else b"\n")
    fp.write(b"  ]")
