"""Streaming-memory acceptance test for #239.

Materializes a synthetic 64 MiB single-member zip through the full
``HttpUrlsPattern`` flow (download -> extract -> hash-compare -> upload)
and asserts that peak Python heap allocation stays within ~32 MiB above
baseline regardless of the extracted member's size.

Pre-#239 the path held the full member as ``bytes`` through hash + upload,
so a 64 MiB member would push peak by 64+ MiB.  Post-#239 the extractor
streams to a tempfile in 8 MiB chunks while hashing in the same pass, and
the upload re-opens the path; peak should track the chunk size, not the
member size.

The test uses 64 MiB rather than the issue's 241 MB because the bound is
what matters: if streaming is broken, a 64 MiB member already blows the
threshold by 4x.  Smaller payload keeps the test fast (~3-5 s) while
still exercising the regression surface.
"""

from __future__ import annotations

import logging
import tracemalloc
import zipfile
from pathlib import Path
from typing import IO, Literal

import httpx
import pytest
import respx

from moncpipelib.contracts.models import IngestContract
from moncpipelib.ingest.patterns.http_urls import HttpUrlsPattern
from moncpipelib.ingest.types import IngestContext, PartitionSpec

# 64 MiB extracted member at 8 MiB write chunks: peak should track the
# 8 MiB chunk, not the 64 MiB total.  Threshold is generous (32 MiB) to
# absorb tracemalloc overhead, httpx framing, and zipfile internals --
# but still less than half the member size, so a regression where the
# full member is buffered would clearly exceed it.
_MEMBER_SIZE_BYTES = 64 * 1024 * 1024
_PEAK_THRESHOLD_BYTES = 32 * 1024 * 1024

# Compressible payload: zlib reduces "a" * N to a few hundred bytes, so
# the on-the-wire zip is tiny but the *extracted* member is full size.
# This is exactly what we want -- the in-memory zip transport doesn't
# inflate the baseline; only the extraction surface does.
_PAYLOAD_BYTE = b"a"


def _build_compressible_zip(zip_path: Path, member_name: str, size_bytes: int) -> None:
    """Stream a `size_bytes`-byte run of ``_PAYLOAD_BYTE`` into a zip member.

    Built chunk-by-chunk so the test setup itself does not buffer the
    full payload into Python memory (which would defeat the baseline).
    """
    chunk = _PAYLOAD_BYTE * (1024 * 1024)  # 1 MiB
    full_chunks, remainder = divmod(size_bytes, len(chunk))
    with (
        zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf,
        zf.open(member_name, "w", force_zip64=True) as out,
    ):
        for _ in range(full_chunks):
            out.write(chunk)
        if remainder:
            out.write(chunk[:remainder])


def _contract() -> IngestContract:
    sensitivity: Literal["public", "confidential", "phi"] = "public"
    return IngestContract(
        source_id="11111111-1111-1111-1111-111111111111",
        source_name="streaming-acceptance",
        sensitivity=sensitivity,
        pattern="http_urls",
        prefix_template="streaming/{partition_key}",
        extract=("zip",),
        strip_extensions=(),
        pattern_config={
            "fetch": {"retries": 0, "timeout_s": 30, "connect_timeout_s": 5},
            "periods": [{"partition_key": "v1", "urls": ["https://upstream.example/big.zip"]}],
        },
    )


class _StreamingFakeBlob:
    """Drains uploads chunk-by-chunk to keep peak memory bounded.

    Records the streamed sha256 + final size so the test can still assert
    the ingest landed correctly.  Does NOT buffer the full upload --
    that would be the very allocation pattern this test pins against.
    """

    def __init__(self) -> None:
        self.uploads: dict[str, tuple[str, int]] = {}

    def read_sha256_metadata(self, sensitivity: str, path: str) -> str | None:
        del sensitivity, path
        return None  # force the upload path

    def upload(
        self,
        sensitivity: str,
        path: str,
        data: bytes | IO[bytes],
        sha256: str,
    ) -> None:
        del sensitivity
        if isinstance(data, bytes):
            size = len(data)
        else:
            size = 0
            for chunk in iter(lambda: data.read(8 * 1024 * 1024), b""):
                size += len(chunk)
        self.uploads[path] = (sha256, size)


@pytest.mark.slow
def test_peak_memory_bounded_for_large_extracted_member(tmp_path: Path) -> None:
    """A 64 MiB extracted member should not blow up peak Python heap.

    Pins the #239 fix: the materialize path streams the member through
    extraction + upload chunk-by-chunk, so peak Python heap above
    baseline tracks the 8 MiB chunk size rather than the member size.
    """
    member_size = _MEMBER_SIZE_BYTES
    zip_path = tmp_path / "big.zip"
    _build_compressible_zip(zip_path, "big.json", member_size)
    on_wire_size = zip_path.stat().st_size
    # Sanity: zlib compresses "a" * N to ~0.1% -- if this assert fails
    # the test setup is the bottleneck, not the extractor.
    assert on_wire_size < member_size // 100, (
        f"compressible zip too large ({on_wire_size} bytes); "
        "transport baseline would dominate the measurement"
    )

    contract = _contract()
    blob = _StreamingFakeBlob()
    ctx = IngestContext(log=logging.getLogger("moncpipelib.test.streaming_memory"))
    spec = PartitionSpec(key="v1", metadata={"partition_key": "v1"})

    # respx.respond requires bytes for `content`; the on-wire zip is
    # ~70 KB thanks to compression so this allocation does not move
    # the peak measurement.  The 64 MiB lives only inside the
    # extracted tempfile.
    zip_bytes = zip_path.read_bytes()
    with respx.mock:
        respx.get("https://upstream.example/big.zip").mock(
            return_value=httpx.Response(200, content=zip_bytes)
        )

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            results = HttpUrlsPattern().materialize_partition(
                contract,
                spec,
                blob,
                ctx,  # type: ignore[arg-type]
            )
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

    # The upload landed and reports the full extracted size.
    assert len(results) == 1
    assert results[0].action == "uploaded"
    assert results[0].size_bytes == member_size
    assert blob.uploads[results[0].path][1] == member_size

    # Peak Python heap stayed within the streaming bound.  If a future
    # change re-introduces full-member buffering, peak will jump to
    # ~member_size and this assertion will fire.
    assert peak <= _PEAK_THRESHOLD_BYTES, (
        f"peak Python heap was {peak / 1024 / 1024:.1f} MiB during a "
        f"{member_size / 1024 / 1024:.0f} MiB ingest -- streaming "
        f"regression?  Threshold: {_PEAK_THRESHOLD_BYTES / 1024 / 1024:.0f} MiB."
    )


# ---------------------------------------------------------------------------
# api_crawl assembly (#415): peak heap pinned to a constant, not record count
# ---------------------------------------------------------------------------

# ~64 MiB of assembled NDJSON from 64k records of ~1 KiB each.  The
# crawl path serializes one record at a time into the hashing tempfile,
# so peak heap should track a single record + json.dumps scratch --
# nowhere near the assembled size.  Threshold mirrors the extractor
# test's "less than half the payload" rule with a large safety margin.
_CRAWL_RECORD_COUNT = 64 * 1024
_CRAWL_RECORD_PAYLOAD = "x" * 1024
_CRAWL_PEAK_THRESHOLD_BYTES = 32 * 1024 * 1024


@pytest.mark.slow
def test_peak_memory_bounded_for_large_crawl_assembly() -> None:
    """#415 acceptance: an api_crawl assembly's peak Python heap is a
    constant over baseline, not a function of record count.

    A plan yielding 64k ~1 KiB records (~64 MiB of NDJSON on disk) must
    not push peak heap anywhere near the assembled size -- records are
    serialized and written to the hashing tempfile one at a time.  If a
    future change accumulates records (e.g. a fold-before-write), peak
    jumps to ~assembly size and this assertion fires.
    """
    from collections.abc import Iterator
    from typing import Any, ClassVar

    from moncpipelib.contracts.models import IngestContract
    from moncpipelib.ingest._throttle import ThrottledClient
    from moncpipelib.ingest.crawl_plans import CRAWL_PLANS, CrawlRecord, register_crawl_plan
    from moncpipelib.ingest.patterns.api_crawl import ApiCrawlPattern

    class _BigCrawlPlan:
        name: ClassVar[str] = "_memory_test_plan"

        def validate_config(self, config: dict[str, Any]) -> list[str]:
            del config
            return []

        def crawl(
            self,
            client: ThrottledClient,
            api_key: str | None,
            config: dict[str, Any],
            ctx: Any,
        ) -> Iterator[CrawlRecord]:
            del client, api_key, config, ctx
            for i in range(_CRAWL_RECORD_COUNT):
                yield CrawlRecord(
                    filename="edges.ndjson",
                    record={"class_id": f"C{i:07d}", "payload": _CRAWL_RECORD_PAYLOAD},
                )

    contract = IngestContract(
        source_id="44444444-4444-4444-4444-444444444444",
        source_name="crawl-memory-acceptance",
        sensitivity="public",
        pattern="api_crawl",
        prefix_template="crawl_mem/{partition_key}",
        extract=(),
        strip_extensions=(),
        pattern_config={
            "crawl_plan": "_memory_test_plan",
            "resolver": "calendar",
            "partition": {"mode": "dynamic", "key_from": "partition_key"},
            # high budget: the stub plan makes no requests, but the
            # ThrottledClient is still constructed from this value.
            "rate_limit_rps": 1000,
        },
    )
    blob = _StreamingFakeBlob()
    ctx = IngestContext(log=logging.getLogger("moncpipelib.test.crawl_memory"))
    spec = PartitionSpec(key="2026-07", metadata={"partition_key": "2026-07"})

    before = dict(CRAWL_PLANS)
    register_crawl_plan(_BigCrawlPlan())
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        results = ApiCrawlPattern().materialize_partition(
            contract,
            spec,
            blob,  # type: ignore[arg-type]
            ctx,
        )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        CRAWL_PLANS.clear()
        CRAWL_PLANS.update(before)

    [result] = results
    assert result.action == "uploaded"
    # each NDJSON line is the sorted-keys record + newline; total lands
    # well above the threshold so the bound is meaningful.
    assert result.size_bytes > _CRAWL_PEAK_THRESHOLD_BYTES
    assert blob.uploads[result.path][1] == result.size_bytes

    assert peak <= _CRAWL_PEAK_THRESHOLD_BYTES, (
        f"peak Python heap was {peak / 1024 / 1024:.1f} MiB while assembling "
        f"{result.size_bytes / 1024 / 1024:.0f} MiB of crawl NDJSON -- "
        f"streaming regression?  Threshold: "
        f"{_CRAWL_PEAK_THRESHOLD_BYTES / 1024 / 1024:.0f} MiB."
    )
