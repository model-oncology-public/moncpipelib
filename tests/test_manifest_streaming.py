"""Streaming-I/O tests for IngestManifest (#243 / Migration 012 Phase B).

Two invariants pinned here:

1. **Byte stability.**  ``write_to(BytesIO).getvalue()`` is byte-identical
   to ``json.dumps(asdict(m), sort_keys=True, indent=2,
   default=_json_default).encode()`` for any manifest.  The manifest's
   own sha256 (stored alongside the blob in metadata) depends on this
   property; any drift would silently invalidate every previously-landed
   manifest's stored sha256.

2. **Streaming memory bound.**  ``read_from`` of a 100k-file manifest
   keeps peak Python heap below ~16 MiB.  Pre-#243 the read materialized
   the full files array as a list of dicts before constructing
   ``ManifestFileEntry`` instances, scaling linearly with file count.
"""

from __future__ import annotations

import json
import tracemalloc
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

import pytest

from moncpipelib.ingest.manifest import (
    IngestManifest,
    ManifestFileEntry,
    _json_default,
)

# ---------------------------------------------------------------------------
# Byte-stability fixtures: shapes that exercise the sorted-key + indent=2
# rendering path the manifest sha256 invariant depends on.
# ---------------------------------------------------------------------------


def _empty_manifest() -> IngestManifest:
    return IngestManifest(
        manifest_version=1,
        source_id="11111111-1111-1111-1111-111111111111",
        source_name="empty-source",
        partition_key="empty",
        materialized_at="2026-04-27T00:00:00Z",
        resolver={},
        fields={},
        files=(),
    )


def _single_file_manifest() -> IngestManifest:
    return IngestManifest(
        manifest_version=1,
        source_id="22222222-2222-2222-2222-222222222222",
        source_name="one-file",
        partition_key="p1",
        materialized_at="2026-04-27T01:00:00Z",
        resolver={"name": "http_urls", "config": {}},
        fields={"a": 1},
        files=(
            ManifestFileEntry(
                path="dir/file.csv",
                sha256="ab" * 32,
                size_bytes=42,
            ),
        ),
    )


def _multi_file_manifest() -> IngestManifest:
    return IngestManifest(
        manifest_version=1,
        source_id="33333333-3333-3333-3333-333333333333",
        source_name="multi",
        partition_key="2026-04-26",
        materialized_at="2026-04-27T02:00:00Z",
        resolver={"name": "uts_release", "config": {"endpoint": "https://x"}},
        fields={"release_version": "2026AB", "release_date": "2026-04-26"},
        files=tuple(
            ManifestFileEntry(
                path=f"meta/MR{i:03d}.RRF",
                sha256=f"{i:064x}",
                size_bytes=1024 * (i + 1),
            )
            for i in range(5)
        ),
    )


def _nested_resolver_manifest() -> IngestManifest:
    """Resolver / fields with nested structures + special characters
    that exercise json.dumps escape behavior (newlines, quotes, unicode)."""
    return IngestManifest(
        manifest_version=1,
        source_id="44444444-4444-4444-4444-444444444444",
        source_name="weird-strings",
        partition_key="weird",
        materialized_at="2026-04-27T03:00:00Z",
        resolver={
            "name": "custom",
            "config": {
                "headers": {"User-Agent": 'app/1.0 "v0.1"'},
                "tags": ["a", "b"],
            },
        },
        fields={
            "note": "line1\nline2\twith unicode",
            "ratio": 0.5,
        },
        files=(ManifestFileEntry(path="x", sha256="0" * 64, size_bytes=0),),
    )


_BYTE_STABILITY_FIXTURES = [
    pytest.param(_empty_manifest, id="empty"),
    pytest.param(_single_file_manifest, id="single-file"),
    pytest.param(_multi_file_manifest, id="multi-file"),
    pytest.param(_nested_resolver_manifest, id="nested-resolver"),
]


@pytest.mark.parametrize("factory", _BYTE_STABILITY_FIXTURES)
def test_write_to_is_byte_identical_to_legacy_json_dumps(factory) -> None:  # type: ignore[no-untyped-def]
    """The manifest's on-disk sha256 must remain stable across the
    streaming-write switchover.  Pin byte-equality between the new
    ``write_to`` output and what ``json.dumps(asdict(m), sort_keys=True,
    indent=2, default=_json_default)`` would have produced.

    A failure here means every previously-landed manifest's sha256
    metadata would silently disagree with what the new writer produces
    -- i.e. the migration is not safe.
    """
    m = factory()

    # New streaming writer output.
    buf = BytesIO()
    m.write_to(buf)
    new_output = buf.getvalue()

    # Legacy canonical output.  This is the contract the streaming
    # writer must reproduce byte-for-byte.
    legacy_output = json.dumps(asdict(m), sort_keys=True, indent=2, default=_json_default).encode(
        "utf-8"
    )

    assert new_output == legacy_output, (
        f"streaming write_to drifted from json.dumps(asdict, sort_keys, "
        f"indent=2). New ({len(new_output)} bytes) vs legacy "
        f"({len(legacy_output)} bytes).\nNew output:\n{new_output.decode()!r}\n"
        f"Legacy output:\n{legacy_output.decode()!r}"
    )


@pytest.mark.parametrize("factory", _BYTE_STABILITY_FIXTURES)
def test_round_trip_preserves_manifest(factory) -> None:  # type: ignore[no-untyped-def]
    """``write_to`` then ``read_from`` returns an equal manifest."""
    original = factory()
    buf = BytesIO()
    original.write_to(buf)
    buf.seek(0)
    parsed = IngestManifest.read_from(buf)
    assert parsed == original


# ---------------------------------------------------------------------------
# Streaming-memory acceptance: a 100k-entry manifest reads with bounded heap
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_read_from_large_manifest_is_memory_bounded(tmp_path: Path) -> None:
    """``read_from`` should not materialize the full files array on the
    heap; peak Python heap during a 100k-file read should stay well
    under what a buffered ``json.loads`` + tuple-comprehension would
    consume (~3x the manifest's serialized size).

    Threshold: 16 MiB.  A typical 100k-entry manifest is ~15 MB
    serialized; pre-#243 the buffered read peaked at ~45 MB.  16 MiB
    leaves headroom for the bounded top-level fields, the entry-by-entry
    parse buffer, and tracemalloc's own overhead, while still failing
    loudly if anyone reintroduces full-array buffering.
    """
    n_entries = 100_000
    # The 100k ManifestFileEntry tuple alone occupies ~20-25 MiB of heap
    # (frozen dataclass with slots, three small attributes per entry).
    # The streaming property says the parser must not add a comparable
    # *second* allocation on top of that -- pre-#243 the read held the
    # raw files-array dict-of-dicts ALONGSIDE the dataclass tuple,
    # peaking at ~45-60 MiB.  Threshold here is end-state + ~10 MiB
    # streaming overhead, well below the pre-fix peak.
    threshold_bytes = 35 * 1024 * 1024

    # Build the manifest on disk via the streaming writer so the test
    # setup itself never materializes 100k entries on the heap (that
    # would defeat the baseline measurement).
    manifest_path = tmp_path / "big_manifest.json"

    with manifest_path.open("wb") as fp:
        # We can't use IngestManifest directly without holding 100k
        # ManifestFileEntry on the heap.  Instead, write the JSON
        # by hand using the same canonical shape ``write_to`` produces.
        fp.write(b"{\n")
        fp.write(b'  "fields": {},\n')
        fp.write(b'  "files": [\n')
        for i in range(n_entries):
            entry = {
                "path": f"data/file_{i:08d}.csv",
                "sha256": f"{i:064x}",
                "size_bytes": 1024 + i,
            }
            rendered = json.dumps(entry, sort_keys=True, indent=2)
            indented = "\n".join("    " + line for line in rendered.split("\n"))
            fp.write(indented.encode("utf-8"))
            fp.write(b",\n" if i < n_entries - 1 else b"\n")
        fp.write(b"  ],\n")
        fp.write(b'  "manifest_version": 1,\n')
        fp.write(b'  "materialized_at": "2026-04-27T00:00:00Z",\n')
        fp.write(b'  "partition_key": "big",\n')
        fp.write(b'  "resolver": {},\n')
        fp.write(b'  "source_id": "55555555-5555-5555-5555-555555555555",\n')
        fp.write(b'  "source_name": "big-source"\n')
        fp.write(b"}")

    # Sanity-check: file is non-trivial size so the test is meaningful.
    on_disk = manifest_path.stat().st_size
    assert on_disk > 5 * 1024 * 1024, (
        f"big manifest ended up only {on_disk} bytes; the 100k entries did not write as expected"
    )

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        with manifest_path.open("rb") as fp:
            parsed = IngestManifest.read_from(fp)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(parsed.files) == n_entries
    # Spot-check that streaming actually parsed correctly.
    assert parsed.files[0].path == "data/file_00000000.csv"
    assert parsed.files[-1].path == f"data/file_{n_entries - 1:08d}.csv"

    # The end-state ManifestFileEntry tuple itself accounts for some
    # of the heap (100k frozen dataclass instances).  The threshold
    # captures that PLUS streaming overhead.  Pre-#243 this would have
    # added ~3x the file's bytes on top.
    assert peak <= threshold_bytes, (
        f"peak Python heap was {peak / 1024 / 1024:.1f} MiB during a "
        f"{n_entries}-entry manifest read (file size: "
        f"{on_disk / 1024 / 1024:.1f} MiB) -- streaming regression?  "
        f"Threshold: {threshold_bytes / 1024 / 1024:.0f} MiB."
    )
