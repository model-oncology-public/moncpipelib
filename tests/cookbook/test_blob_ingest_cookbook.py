"""Cookbook tests for the universal blob-landing ingest boundary.

Code between ``# --- cookbook:start ---`` / ``# --- cookbook:end ---``
is extracted into ``docs/cookbook.md`` by the cookbook plugin.

The example uses a tiny in-memory stand-in for ``BlobStorageResource``
so it runs deterministically in CI. Real pipelines configure
``BlobStorageResource`` against their actual storage account.
"""

from __future__ import annotations

import pytest


@pytest.mark.cookbook(
    title="Land an HTTP source with http_urls + resolve blob refs downstream",
    description=(
        "Phase 1+2 of the universal blob-landing ingest boundary. Declare an "
        "``IngestContract`` for the http_urls pattern, land data once per "
        "partition via ``materialize_with_manifest`` (the canonical entry "
        "point as of v0.26.0), and then resolve the landed blob from a "
        "downstream ``DataSource`` via ``resolve_source_for_partition``. "
        "The dispatcher writes a per-partition ``_manifest.json`` so consumer "
        "drift detection works end-to-end. The in-memory blob stand-in keeps "
        "the example deterministic; real pipelines inject ``BlobStorageResource``."
    ),
    category="ingest",
)
def test_cookbook_http_urls_roundtrip() -> None:
    # --- cookbook:start ---
    import hashlib
    import io
    import logging
    import zipfile
    from collections.abc import Iterator
    from datetime import date
    from typing import IO

    import respx

    from moncpipelib.contracts.models import (
        ContractCorpus,
        DataSource,
        IngestContract,
        Period,
    )
    from moncpipelib.ingest import (
        BlobRef,
        HttpUrlsPattern,
        IngestContext,
        PartitionSpec,
        materialize_with_manifest,
        resolve_source_for_partition,
    )

    # --- 1. Declare the ingest contract (normally loaded from a *.ingest.yaml) ---
    ingest = IngestContract(
        source_id="11111111-1111-1111-1111-111111111111",
        source_name="vendor-prices",
        sensitivity="public",
        pattern="http_urls",
        prefix_template="vendor_prices/{partition_key}",
        extract=("zip",),
        strip_extensions=(),
        pattern_config={
            # fetch.user_agent (optional, #413): a descriptive organizational
            # UA for upstreams that reject default script User-Agents
            # (e.g. FDA accessdata abuse detection). Omit to send httpx's
            # default. Applies to the payload download of both http_urls
            # and api_resolver; resolver probes set their own headers.
            "fetch": {
                "retries": 0,
                "timeout_s": 5,
                "user_agent": "ExampleOrgDataPlatform/1.0 (contact: data@example.org)",
            },
            "periods": [
                {
                    "partition_key": "2024-q1",
                    "urls": ["https://upstream.example/2024-q1.zip"],
                },
            ],
        },
    )

    # --- 2. Declare the downstream source (normally loaded from a *.source.yaml) ---
    source = DataSource(
        source_id="22222222-2222-2222-2222-222222222222",
        source_name="vendor-prices-extract",
        periods=[
            Period(
                source="*prices*.csv",  # glob, not URL
                effective_from=date(2024, 1, 1),
                partition_key="2024-q1",
            )
        ],
        ingest_source="vendor-prices",  # link to the ingest contract
    )

    corpus = ContractCorpus(
        ingests={ingest.source_name: ingest},
        sources={source.source_name: source},
    )

    # --- 3. Tiny in-memory blob stand-in so the example runs in CI ---
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
            # The pattern hands a streaming file handle for large members
            # (#239); accept either bytes or IO[bytes] for the in-memory fake.
            body = data if isinstance(data, bytes) else data.read()
            self.store[path] = (body, sha256)

        def exists(self, sensitivity: str, path: str) -> bool:
            del sensitivity
            return path in self.store

        def download(self, sensitivity: str, path: str) -> bytes:
            del sensitivity
            return self.store[path][0]

    blob = InMemoryBlob()

    # --- 4. Build the IngestContext (logger + optional secrets) ---
    # http_urls does not need ctx.secrets; api_resolver does.
    ctx = IngestContext(log=logging.getLogger("ingest-cookbook"))

    # --- 5. Materialize via materialize_with_manifest ---
    # This is the canonical entry point as of v0.26.0. It calls the
    # pattern's materialize_partition and then atomically writes
    # {prefix}/_manifest.json so consumer-side drift detection sees a
    # complete partition or no partition at all.
    def _zip_bytes(files: dict[str, bytes]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, data in files.items():
                zf.writestr(name, data)
        return buf.getvalue()

    file_bytes = b"sku,price\nA-001,10.00\n"
    payload = _zip_bytes({"prices.csv": file_bytes})

    with respx.mock:
        respx.get("https://upstream.example/2024-q1.zip").respond(200, content=payload)

        results = materialize_with_manifest(
            HttpUrlsPattern(),
            ingest,
            PartitionSpec(key="2024-q1"),
            blob,  # type: ignore[arg-type]
            ctx,
        )

    # First run uploads the data file AND writes _manifest.json.
    assert all(r.action == "uploaded" for r in results)
    expected_path = "vendor_prices/2024-q1/prices.csv"
    assert expected_path in blob.store
    assert "vendor_prices/2024-q1/_manifest.json" in blob.store

    # --- 6. Resolve the landed blob from the downstream source ---
    # The static-period branch (enumerated periods) ignores the manifest;
    # it just glob-matches under the prefix. The FromIngestTemplate branch
    # reads the manifest -- see the api_resolver cookbook for that flow.
    [ref] = resolve_source_for_partition(
        source,
        partition_key="2024-q1",
        corpus=corpus,
        blob=blob,  # type: ignore[arg-type]
    )
    assert isinstance(ref, BlobRef)
    assert ref.path == expected_path
    assert ref.sensitivity == "public"

    # --- 7. Re-materialize: idempotent skip via sha256 header ---
    assert (
        blob.read_sha256_metadata("public", expected_path) == hashlib.sha256(file_bytes).hexdigest()
    )

    with respx.mock:
        respx.get("https://upstream.example/2024-q1.zip").respond(200, content=payload)
        second_run = materialize_with_manifest(
            HttpUrlsPattern(),
            ingest,
            PartitionSpec(key="2024-q1"),
            blob,  # type: ignore[arg-type]
            ctx,
        )

    assert all(r.action == "skipped" for r in second_run)
    # --- cookbook:end ---
