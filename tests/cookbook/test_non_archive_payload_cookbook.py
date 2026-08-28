"""Cookbook: non-archive http_urls ingest with descriptive consumer glob.

Demonstrates the URL-basename precedence chain landed in #270.  Before
this change, ``extract: []`` contracts forced consumers to glob for
the literal ``"__payload__"``.  Now the body lands under the URL's
basename (or ``payload_filename_template`` when set), and consumers
glob descriptively as they would for any archive contract.
"""

from __future__ import annotations

import pytest


@pytest.mark.cookbook(
    title="Non-archive http_urls ingest with descriptive consumer glob",
    description=(
        "Land a single CSV (no zip) via the ``http_urls`` pattern with "
        "``extract: []``.  The body lands under the URL's sanitized "
        "basename, so the downstream consumer glob can use a descriptive "
        "pattern like ``*.csv`` rather than the legacy ``__payload__`` "
        "literal.  When the URL's basename is ambiguous, set "
        "``payload_filename_template`` on the contract for an explicit "
        "name -- the template is rendered with ``{partition_key}`` and "
        "``{source_name}``."
    ),
    category="ingest",
)
def test_cookbook_non_archive_payload_filename() -> None:
    # --- cookbook:start ---
    import io
    import logging
    import zipfile  # noqa: F401  -- kept for symmetry with archive cookbook
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

    # --- 1. Declare a non-archive ingest contract ---
    # ``extract: ()`` (an empty tuple) tells the pattern there is no
    # archive to expand: the response body is the data file.  No
    # ``payload_filename_template`` is set, so the URL basename
    # ("V2024B_V2025B_V2026A_CPC_SMVL.csv") becomes the landed name.
    ingest = IngestContract(
        source_id="11111111-1111-1111-1111-111111111111",
        source_name="seer-cpc-smvl",
        sensitivity="public",
        pattern="http_urls",
        prefix_template="seer_cpc_smvl/{partition_key}",
        extract=(),  # NON-ARCHIVE
        strip_extensions=(),
        pattern_config={
            "fetch": {"retries": 0, "timeout_s": 5},
            "periods": [
                {
                    "partition_key": "V2024B_V2025B_V2026A",
                    "urls": [
                        "https://seer.example/files/V2024B_V2025B_V2026A_CPC_SMVL.csv",
                    ],
                },
            ],
        },
    )

    # --- 2. Downstream consumer uses a descriptive glob ---
    # Pre-#270 this had to be ``"__payload__"`` because that was the
    # literal helper-emitted name.  Now any glob that matches the URL
    # basename works -- ``*.csv`` is sufficient for single-file periods.
    source = DataSource(
        source_id="22222222-2222-2222-2222-222222222222",
        source_name="seer-cpc-smvl-bronze",
        periods=[
            Period(
                source="*_CPC_SMVL.csv",
                effective_from=date(2026, 5, 1),
                partition_key="V2024B_V2025B_V2026A",
            )
        ],
        ingest_source="seer-cpc-smvl",
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
            del sensitivity
            return (p for p in self.store if p.startswith(prefix))

        def read_sha256_metadata(self, sensitivity: str, path: str) -> str | None:
            del sensitivity
            entry = self.store.get(path)
            return entry[1] if entry else None

        def upload(self, sensitivity: str, path: str, data: bytes | IO[bytes], sha256: str) -> None:
            del sensitivity
            body = data if isinstance(data, bytes) else data.read()
            self.store[path] = (body, sha256)

        def exists(self, sensitivity: str, path: str) -> bool:
            del sensitivity
            return path in self.store

        def download(self, sensitivity: str, path: str) -> bytes:
            del sensitivity
            return self.store[path][0]

        def stream(self, sensitivity: str, path: str) -> IO[bytes]:
            del sensitivity
            return io.BytesIO(self.store[path][0])

    blob = InMemoryBlob()
    ctx = IngestContext(log=logging.getLogger("non-archive-cookbook"))

    # --- 4. Materialize: body lands under the URL basename ---
    file_bytes = b"site,histology,behavior,validity\nC50,8500/3,3,valid\n"

    with respx.mock:
        respx.get("https://seer.example/files/V2024B_V2025B_V2026A_CPC_SMVL.csv").respond(
            200, content=file_bytes
        )

        materialize_with_manifest(
            HttpUrlsPattern(),
            ingest,
            PartitionSpec(key="V2024B_V2025B_V2026A"),
            blob,  # type: ignore[arg-type]
            ctx,
        )

    # The body lands under its URL basename, NOT __payload__.
    expected_path = "seer_cpc_smvl/V2024B_V2025B_V2026A/V2024B_V2025B_V2026A_CPC_SMVL.csv"
    assert expected_path in blob.store

    # --- 5. The descriptive consumer glob resolves cleanly ---
    [ref] = resolve_source_for_partition(
        source,
        partition_key="V2024B_V2025B_V2026A",
        corpus=corpus,
        blob=blob,  # type: ignore[arg-type]
    )
    assert isinstance(ref, BlobRef)
    assert ref.path == expected_path
    # --- cookbook:end ---
