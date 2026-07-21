"""Cookbook tests for the ``calendar`` release resolver (per #218).

The calendar resolver synthesizes partition keys from a calendar
cadence (weekly / monthly / quarterly) without making any upstream
API call. Pairs with ``api_resolver`` to support sources whose
download URL is stable across snapshots and whose partition cadence
is consumer-defined.

Code between ``# --- cookbook:start ---`` / ``# --- cookbook:end ---``
is extracted into ``docs/cookbook.md`` by the cookbook plugin.
"""

from __future__ import annotations

import pytest
from freezegun import freeze_time


@pytest.mark.cookbook(
    title="Land a rolling-URL source with the calendar resolver",
    description=(
        "The ``calendar`` resolver synthesizes weekly / monthly / quarterly "
        "partition keys without any upstream API call. Use it for sources "
        "whose download URL is stable (one rolling URL) but where you still "
        "want per-snapshot history -- consumer-defined cadence. The example "
        "declares a weekly Sunday-anchored cadence, lands data via "
        "``materialize_with_manifest``, and re-materializes to demonstrate "
        "``hash_compare`` idempotency. No ``credential`` block is needed: "
        "the calendar resolver doesn't authenticate (per #218)."
    ),
    category="ingest",
)
@freeze_time("2026-04-22 12:00:00")  # Wednesday -> prior Sunday partition
def test_cookbook_calendar_resolver_roundtrip() -> None:
    # --- cookbook:start ---
    import io
    import logging
    import zipfile
    from collections.abc import Iterator
    from typing import IO

    import respx

    from moncpipelib.contracts.models import (
        ContractCorpus,
        DataSource,
        FromIngestTemplate,
        IngestContract,
    )
    from moncpipelib.ingest import (
        ApiResolverPattern,
        BlobRef,
        IngestContext,
        materialize_with_manifest,
        resolve_source_for_partition,
    )

    # --- 1. Declare the api_resolver ingest contract using the calendar resolver ---
    # No `credential` block: the calendar resolver synthesizes the partition
    # key from the configured cadence and doesn't authenticate.
    ingest = IngestContract(
        source_id="11111111-1111-1111-1111-111111111111",
        source_name="vendor-rolling-feed",
        sensitivity="public",
        pattern="api_resolver",
        prefix_template="vendor_rolling/{partition_key}",
        extract=("zip",),
        strip_extensions=(),
        pattern_config={
            "resolver": "calendar",
            "resolver_config": {
                "start_date": "2024-01-01",
                "cadence": "weekly",
                "anchor_dow": "Sunday",
                "anchor_tz": "UTC",
                "url": "https://upstream.example/rolling.zip",
            },
            "partition": {"mode": "dynamic", "key_from": "snapshot_date"},
            "idempotency": "hash_compare",
            "fetch": {"retries": 0, "timeout_s": 5},
        },
    )

    # --- 2. Declare the downstream source (FromIngestTemplate) ---
    source = DataSource(
        source_id="22222222-2222-2222-2222-222222222222",
        source_name="vendor-rolling-extract",
        periods=FromIngestTemplate(
            source="data.csv",
            effective_from_field="snapshot_date",
        ),
        ingest_source="vendor-rolling-feed",
    )

    corpus = ContractCorpus(
        ingests={ingest.source_name: ingest},
        sources={source.source_name: source},
    )

    # --- 3. In-memory blob stand-in ---
    class InMemoryBlob:
        def __init__(self) -> None:
            self.store: dict[str, tuple[bytes, str]] = {}

        def list(self, sensitivity: str, prefix: str) -> list[str]:
            del sensitivity
            return [p for p in self.store if p.startswith(prefix)]

        def iter_list(self, sensitivity: str, prefix: str) -> Iterator[str]:
            # Lazy iterator (#246) -- consumers prefer this for large prefixes.
            del sensitivity
            return (p for p in self.store if p.startswith(prefix))

        def read_sha256_metadata(self, sensitivity: str, path: str) -> str | None:
            del sensitivity
            entry = self.store.get(path)
            return entry[1] if entry else None

        def upload(self, sensitivity: str, path: str, data: bytes | IO[bytes], sha256: str) -> None:
            del sensitivity
            # The pattern + dispatcher stream uploads from a file handle
            # for large members and the manifest (#239 / #243); accept
            # either bytes or IO[bytes] in the fake.
            body = data if isinstance(data, bytes) else data.read()
            self.store[path] = (body, sha256)

        def exists(self, sensitivity: str, path: str) -> bool:
            del sensitivity
            return path in self.store

        def download(self, sensitivity: str, path: str) -> bytes:
            del sensitivity
            return self.store[path][0]

        def stream(self, sensitivity: str, path: str) -> IO[bytes]:
            # Forward-only file-like for the streaming manifest read
            # path (#241 / #243).
            del sensitivity
            return io.BytesIO(self.store[path][0])

    blob = InMemoryBlob()

    # --- 4. Build the IngestContext ---
    # No secrets resource: the calendar resolver doesn't authenticate.
    ctx = IngestContext(log=logging.getLogger("calendar-cookbook"))

    # --- 5. Discover the current partition ---
    # Frozen time is a Wednesday; weekly Sunday-anchored cadence picks
    # the prior Sunday (2026-04-19) as the partition key.
    pattern = ApiResolverPattern()
    [partition_spec] = pattern.discover_partitions(ingest, ctx)
    assert partition_spec.key == "2026-04-19"
    assert partition_spec.metadata["snapshot_date"] == "2026-04-19"

    # --- 6. Materialize via materialize_with_manifest ---
    def _zip_bytes(files: dict[str, bytes]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, data in files.items():
                zf.writestr(name, data)
        return buf.getvalue()

    payload = _zip_bytes({"data.csv": b"id,value\n1,a\n"})

    with respx.mock:
        respx.get("https://upstream.example/rolling.zip").respond(200, content=payload)

        results = materialize_with_manifest(
            pattern,
            ingest,
            partition_spec,
            blob,  # type: ignore[arg-type]
            ctx,
        )

    assert all(r.action == "uploaded" for r in results)
    assert "vendor_rolling/2026-04-19/data.csv" in blob.store
    assert "vendor_rolling/2026-04-19/_manifest.json" in blob.store

    # --- 7. Resolve via the FromIngestTemplate branch (manifest reader) ---
    [ref] = resolve_source_for_partition(
        source,
        partition_key="2026-04-19",
        corpus=corpus,
        blob=blob,  # type: ignore[arg-type]
    )
    assert isinstance(ref, BlobRef)
    assert ref.path == "vendor_rolling/2026-04-19/data.csv"

    # --- 8. Re-materialize: idempotent skip via hash_compare ---
    # Upstream returns the same bytes; the calendar resolver returns the
    # same partition key for the same week. Re-uploading is skipped.
    with respx.mock:
        respx.get("https://upstream.example/rolling.zip").respond(200, content=payload)
        second_run = materialize_with_manifest(
            pattern,
            ingest,
            partition_spec,
            blob,  # type: ignore[arg-type]
            ctx,
        )

    data_results = [r for r in second_run if r.path.endswith(".csv")]
    assert all(r.action == "skipped" for r in data_results)
    # --- cookbook:end ---
