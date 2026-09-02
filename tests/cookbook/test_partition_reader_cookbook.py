"""Cookbook entry for :func:`read_partition_with_manifest`.

Code between ``# --- cookbook:start ---`` / ``# --- cookbook:end ---``
is extracted into ``docs/cookbook.md`` by the cookbook plugin.

The example uses a tiny in-memory blob stand-in so it runs deterministically
in CI.  Real bronze pipelines inject ``BlobStorageResource`` and an
``IngestContract`` / ``DataSource`` loaded from the project's contract
corpus.
"""

from __future__ import annotations

import pytest


@pytest.mark.cookbook(
    title="Read a partition's blob and manifest in one call",
    description=(
        "Bronze pipelines fed by an ``api_resolver`` ingest contract need "
        "both the blob's bytes and a few fields out of its manifest "
        "(``release_date``, ``release_version``, etc.).  ``read_partition_"
        "with_manifest`` collapses the five-step boilerplate (resolve, "
        "type-check, open, load manifest, validate fields) into a single "
        "context manager.  The blob is streamed -- peak memory is bounded "
        "by the blob library's chunk size regardless of file size -- and "
        "the manifest is parsed via ``IngestManifest.read_from`` so the "
        "forward-compat ``manifest_version`` check is preserved."
    ),
    category="ingest",
)
def test_cookbook_read_partition_with_manifest() -> None:
    # --- cookbook:start ---
    import io
    from datetime import date
    from typing import IO

    from moncpipelib.contracts.models import (
        ContractCorpus,
        DataSource,
        FromIngestTemplate,
        IngestContract,
    )
    from moncpipelib.ingest import drain_to_bytes, read_partition_with_manifest

    # --- 1. Contracts (normally loaded from *.ingest.yaml + *.source.yaml) ---
    ingest = IngestContract(
        source_id="11111111-1111-1111-1111-111111111111",
        source_name="seer-cpc",
        sensitivity="confidential",
        pattern="api_resolver",
        prefix_template="seer_cpc/{partition_key}",
        extract=(),
        strip_extensions=(),
        pattern_config={
            "resolver": "seer_release",
            "resolver_config": {},
            "credential": {"secret_name": "seer-api-key"},
            "partition": {"mode": "dynamic", "key_from": "release_version"},
        },
    )
    source = DataSource(
        source_id="22222222-2222-2222-2222-222222222222",
        source_name="seer-cpc-bronze",
        periods=FromIngestTemplate(
            source="*.csv",
            effective_from_field="release_date",
        ),
        ingest_source="seer-cpc",
    )
    corpus = ContractCorpus(
        ingests={ingest.source_name: ingest},
        sources={source.source_name: source},
    )

    # --- 2. Tiny in-memory blob stand-in so the example runs in CI ---
    # The production ``BlobStorageResource.stream`` returns a
    # forward-only, *bounded-only* reader: ``read()`` with no size
    # argument raises (it would materialize the whole blob and negate
    # the streaming bound).  The stand-in mirrors that contract so the
    # example stays honest -- a plain ``io.BytesIO`` would silently
    # accept ``read()`` and let a broken example pass in CI.
    class _BoundedReader(io.RawIOBase):
        def __init__(self, data: bytes) -> None:
            super().__init__()
            self._buf = io.BytesIO(data)

        def readable(self) -> bool:
            return True

        def readinto(self, b: bytearray) -> int:  # type: ignore[override]
            return self._buf.readinto(b)

        def readall(self) -> bytes:
            raise io.UnsupportedOperation(
                "Unbounded read would materialize the full blob; use read(n) or drain_to_bytes()."
            )

    class InMemoryBlob:
        def __init__(self, contents: dict[str, bytes]) -> None:
            self.contents = dict(contents)

        def iter_list(self, sensitivity: str, prefix: str):  # noqa: ANN201
            del sensitivity
            return (p for p in self.contents if p.startswith(prefix))

        def exists(self, sensitivity: str, path: str) -> bool:
            del sensitivity
            return path in self.contents

        def stream(self, sensitivity: str, path: str) -> IO[bytes]:
            del sensitivity
            return _BoundedReader(self.contents[path])  # type: ignore[return-value]

    manifest_json = (
        "{\n"
        '  "fields": {"release_date": "2024-09-01", "release_version": "V2024B"},\n'
        '  "files": [{"path": "seer_cpc/V2024B/data.csv", "sha256": "abc", "size_bytes": 12}],\n'
        '  "manifest_version": 1,\n'
        '  "materialized_at": "2024-09-01T12:00:00Z",\n'
        '  "partition_key": "V2024B",\n'
        '  "resolver": {"config": {}, "name": "seer_release"},\n'
        '  "source_id": "11111111-1111-1111-1111-111111111111",\n'
        '  "source_name": "seer-cpc"\n'
        "}"
    )
    blob = InMemoryBlob(
        contents={
            "seer_cpc/V2024B/_manifest.json": manifest_json.encode("utf-8"),
            "seer_cpc/V2024B/data.csv": b"id,value\n1,42\n",
        },
    )

    # --- 3. Read the blob + manifest fields in one call ---
    # The context manager yields (ref, blob_stream, manifest_fields).
    # ``required_fields`` raises ``ManifestFieldError`` (a subclass of
    # ``IngestResolutionError``) if any name is absent or empty, with the
    # full sorted list of available fields in the message.
    with read_partition_with_manifest(
        source=source,
        partition_key="V2024B",
        corpus=corpus,
        blob=blob,  # type: ignore[arg-type]
        required_fields=("release_date", "release_version"),
    ) as (ref, blob_stream, manifest_fields):
        # ``ref`` is a BlobRef -- use it for logging / lineage.
        assert ref.sensitivity == "confidential"

        # ``blob_stream`` is a forward-only, *bounded-only* IO[bytes]:
        # ``blob_stream.read()`` with no size argument raises (and so
        # does handing the stream straight to polars / calamine, which
        # fall through to that unbounded read internally).  For a small
        # reference file, drain it with an explicit ceiling, then wrap
        # the bytes in BytesIO for whichever parser wants seekable input
        # (polars: ``pl.read_csv(buf)``; pandas: ``pd.read_csv(buf)``):
        buf = io.BytesIO(drain_to_bytes(blob_stream, max_bytes=8 * 1024 * 1024))
        csv_bytes = buf.getvalue()
        # For a payload too large to hold in memory, skip this helper and
        # use ``BlobStorageResource.download_to_path(...)`` to stream the
        # blob to local disk, then open the on-disk file.

        # ``manifest_fields`` is ``manifest.fields`` verbatim -- callers
        # do their own date parsing / type coercion.
        release_date = date.fromisoformat(manifest_fields["release_date"])
        release_version = manifest_fields["release_version"]

    assert csv_bytes == b"id,value\n1,42\n"
    assert release_date == date(2024, 9, 1)
    assert release_version == "V2024B"
    # --- cookbook:end ---
