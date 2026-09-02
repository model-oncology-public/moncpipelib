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


# ---------------------------------------------------------------------------
# browser_export (#463): peak Python heap pinned to a constant, not download
# size; peak on-disk footprint pinned to a bounded multiple, not a single
# copy (D9's scoped exception to the I/O-at-Boundaries rule)
# ---------------------------------------------------------------------------

# 64 MiB downloaded file: peak heap should track the read-back chunk size
# (_HASH_CHUNK_BYTES = 1 MiB) plus the upload's own chunking, nowhere near
# the download size.
_BROWSER_EXPORT_MEMBER_SIZE_BYTES = 64 * 1024 * 1024
_BROWSER_EXPORT_PEAK_THRESHOLD_BYTES = 32 * 1024 * 1024


def _write_large_file(path: Path, size_bytes: int) -> None:
    """Stream a `size_bytes`-byte run of ``b"a"`` to ``path`` in 1 MiB chunks.

    Mirrors ``_build_compressible_zip`` above: built chunk-by-chunk so the
    setup itself does not buffer the full payload into Python memory.
    """
    chunk = b"a" * (1024 * 1024)
    full_chunks, remainder = divmod(size_bytes, len(chunk))
    with path.open("wb") as fp:
        for _ in range(full_chunks):
            fp.write(chunk)
        if remainder:
            fp.write(chunk[:remainder])


@pytest.mark.slow
def test_peak_memory_bounded_for_large_browser_export_download(tmp_path: Path) -> None:
    """#463 acceptance: browser_export's read-back hash pass + upload pass
    keep peak Python heap bounded by chunk size, not download size.

    A stubbed session's ``click_and_await_download`` writes a 64 MiB file
    (chunk-by-chunk, so setup doesn't inflate the baseline) and returns an
    ``ExportedFile`` pointing at it -- standing in for what a real
    playwright ``Download.save_as`` would have already produced on disk
    before ``BrowserExportPattern.materialize_partition`` ever sees it.
    """
    import logging as _logging
    from contextlib import contextmanager
    from typing import Any

    from freezegun import freeze_time

    from moncpipelib.contracts.models import IngestContract
    from moncpipelib.ingest.export_plans import EXPORT_PLANS, ExportedFile, register_export_plan
    from moncpipelib.ingest.patterns import browser_export as browser_export_module
    from moncpipelib.ingest.patterns.browser_export import BrowserExportPattern
    from moncpipelib.ingest.types import IngestContext, PartitionSpec

    member_size = _BROWSER_EXPORT_MEMBER_SIZE_BYTES
    download_path = tmp_path / "0000.download"
    _write_large_file(download_path, member_size)

    class _FakeDownloadSession:
        def click_and_await_download(self, **kwargs: Any) -> ExportedFile:
            del kwargs
            return ExportedFile(path=download_path, suggested_filename="big_export.csv")

        def require_contains(self, path: Any) -> None:
            del path

    @contextmanager
    def _fake_browser_session(**kwargs: Any):  # type: ignore[no-untyped-def]
        del kwargs
        yield _FakeDownloadSession()

    class _BigBrowserExportPlan:
        name = "_memory_test_browser_export_plan"

        def validate_config(self, config: dict[str, Any]) -> list[str]:
            del config
            return []

        def export(self, session: Any, partition_key: str, config: dict[str, Any], ctx: Any) -> Any:
            del partition_key, config, ctx
            yield session.click_and_await_download(role="button", name="Export")

    contract = IngestContract(
        source_id="66666666-6666-6666-6666-666666666666",
        source_name="browser-export-memory-acceptance",
        sensitivity="public",
        pattern="browser_export",
        prefix_template="browser_mem/{partition_key}",
        extract=(),
        strip_extensions=(),
        pattern_config={
            "export_plan": "_memory_test_browser_export_plan",
            "export_config": {},
            "allowed_hosts": ["340bopais.hrsa.gov"],
            "partition": {"mode": "dynamic", "cadence": "daily", "anchor_tz": "UTC"},
        },
        compliance_review="SECURITY.md#memory-acceptance",
    )
    blob = _StreamingFakeBlob()
    ctx = IngestContext(log=_logging.getLogger("moncpipelib.test.browser_export_memory"))

    before_plans = dict(EXPORT_PLANS)
    register_export_plan(_BigBrowserExportPlan())
    real_browser_session = browser_export_module.browser_session
    browser_export_module.browser_session = _fake_browser_session  # type: ignore[assignment]
    try:
        pattern = BrowserExportPattern()
        with freeze_time("2026-08-06 12:00:00"):
            spec = PartitionSpec(key="2026-08-06", metadata={})
            tracemalloc.start()
            try:
                tracemalloc.reset_peak()
                results = pattern.materialize_partition(
                    contract,
                    spec,
                    blob,  # type: ignore[arg-type]
                    ctx,
                )
                _current, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
    finally:
        EXPORT_PLANS.clear()
        EXPORT_PLANS.update(before_plans)
        browser_export_module.browser_session = real_browser_session  # type: ignore[assignment]

    assert len(results) == 1
    assert results[0].action == "uploaded"
    assert results[0].size_bytes == member_size
    assert blob.uploads[results[0].path][1] == member_size

    assert peak <= _BROWSER_EXPORT_PEAK_THRESHOLD_BYTES, (
        f"peak Python heap was {peak / 1024 / 1024:.1f} MiB during a "
        f"{member_size / 1024 / 1024:.0f} MiB browser_export download -- "
        f"streaming regression?  Threshold: "
        f"{_BROWSER_EXPORT_PEAK_THRESHOLD_BYTES / 1024 / 1024:.0f} MiB."
    )


_DISK_FOOTPRINT_MEMBER_SIZE_BYTES = 8 * 1024 * 1024
_DISK_FOOTPRINT_MULTIPLE = 2.0
"""Peak on-disk footprint under the session tempdir, as a multiple of the
downloaded payload's own size.

Named and pinned at exactly 2x -- NOT 1x -- because ``Download.save_as``
was verified empirically (#463 Step 2, against a real chromium 1.62.0) to
COPY the artifact rather than move it: playwright retains its own
internally-managed copy under ``downloads_path`` until the browser session
closes, and ``save_as`` writes a second, independent copy at the
destination path. ``browser_session()`` points ``chromium.launch(...,
downloads_path=...)`` at its own per-run tempdir specifically so BOTH
copies land under the one directory this test (and the session's own
cleanup) can account for -- see ``ingest/_browser.py``'s module docstring
"Disk footprint, stated honestly" section. A correct implementation must
never exceed this multiple (the pattern itself performs no additional
on-disk copy of the payload -- see the module docstring's "pass count over
the payload" note); a regression that reintroduces a third copy (e.g. a
``hashing_tempfile`` re-buffer) would push this past 2x and this assertion
would fire.
"""


@pytest.mark.slow
def test_peak_disk_footprint_bounded_and_session_tempdir_cleared_on_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D9's disk-footprint claim, corrected (#463 Step 3 prerequisite B).

    The originally-specified test asserted ``<= 1.1 * payload_size`` --
    i.e. approximately one copy on disk. That is FALSE as specified:
    Step 2 established empirically that ``Download.save_as`` copies
    rather than moves, so peak on-disk footprint under the session's own
    tempdir is approximately **2x** the payload size (playwright's own
    retained artifact plus ``save_as``'s copy), not 1x. This test asserts
    the invariant that IS true and worth protecting instead: peak disk is
    bounded by :data:`_DISK_FOOTPRINT_MULTIPLE` regardless of payload
    size (independent of size, like the peak-heap pin above), and the
    session's tempdir is fully removed after ``browser_session()``'s
    ``with`` block exits -- on both a successful materialization and one
    that fails mid-upload.

    Drives the REAL ``browser_session()`` context manager (not
    monkeypatched away, unlike the other browser_export tests in this
    repo) against a fake playwright module, so the tempdir-cleanup
    guarantee under test is the genuine ``ExitStack`` unwind, not a
    stand-in for it.
    """
    import sys
    import types
    from typing import Any

    from freezegun import freeze_time

    from moncpipelib.contracts.models import IngestContract
    from moncpipelib.ingest import _browser
    from moncpipelib.ingest.export_plans import EXPORT_PLANS, register_export_plan
    from moncpipelib.ingest.patterns.browser_export import BrowserExportPattern
    from moncpipelib.ingest.types import IngestContext, PartitionSpec

    member_size = _DISK_FOOTPRINT_MEMBER_SIZE_BYTES

    class _FakeDownload:
        def __init__(self, tempdir_capture: dict[str, Path]) -> None:
            self.suggested_filename = "big_export.csv"
            self._tempdir_capture = tempdir_capture

        def save_as(self, dest: Path) -> None:
            # dest.parent is the session's `downloads/` dir; its parent
            # is the per-run tempdir root browser_session() created.
            self._tempdir_capture["path"] = dest.parent.parent
            _write_large_file(dest, member_size)
            # Model playwright's own retained copy (see
            # _DISK_FOOTPRINT_MULTIPLE's docstring) -- a fake that only
            # ever wrote the save_as target would understate the real
            # ~2x footprint and this test would pin nothing.
            _write_large_file(dest.parent / f".retained-{dest.name}", member_size)

        def failure(self) -> str | None:
            return None

    class _FakeLocator:
        def click(self, *, timeout: float) -> None:
            del timeout

    class _FakeExpectDownloadCM:
        def __init__(self, download: Any) -> None:
            self._download = download
            self.value: Any = None

        def __enter__(self) -> Any:
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
            if exc_val is not None:
                return False
            self.value = self._download
            return False

    class _FakePage:
        def __init__(self, download: Any) -> None:
            self._locator = _FakeLocator()
            self._download = download

        def get_by_role(self, role: str, *, name: str, exact: bool) -> _FakeLocator:
            del role, name, exact
            return self._locator

        def expect_download(self, *, timeout: float) -> _FakeExpectDownloadCM:
            del timeout
            return _FakeExpectDownloadCM(self._download)

        def set_default_timeout(self, timeout: float) -> None:
            del timeout

        def set_default_navigation_timeout(self, timeout: float) -> None:
            del timeout

    class _FakeContext:
        def __init__(self, page: _FakePage) -> None:
            self._page = page
            self.closed = False

        def route(self, pattern: str, handler: Any) -> None:
            del pattern, handler

        def route_web_socket(self, pattern: str, handler: Any) -> None:
            del pattern, handler

        def new_page(self) -> _FakePage:
            return self._page

        def close(self) -> None:
            self.closed = True

    class _FakeBrowser:
        def __init__(self, context: _FakeContext) -> None:
            self._context = context
            self.closed = False

        def new_context(self, **kwargs: Any) -> _FakeContext:
            del kwargs
            return self._context

        def close(self) -> None:
            self.closed = True

    class _FakeChromium:
        def __init__(self, browser: _FakeBrowser) -> None:
            self._browser = browser

        def launch(self, **kwargs: Any) -> _FakeBrowser:
            del kwargs
            return self._browser

    class _FakePwHandle:
        def __init__(self, chromium: _FakeChromium) -> None:
            self.chromium = chromium

    class _FakeSyncPlaywrightCM:
        def __init__(self, chromium: _FakeChromium) -> None:
            self._chromium = chromium

        def __enter__(self) -> Any:
            return _FakePwHandle(self._chromium)

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
            return False

    def _install_fake_playwright(tempdir_capture: dict[str, Path]) -> None:
        monkeypatch.setattr(_browser, "ensure_chromium_available", lambda: None)
        download = _FakeDownload(tempdir_capture)
        page = _FakePage(download)
        context = _FakeContext(page)
        browser = _FakeBrowser(context)
        chromium = _FakeChromium(browser)
        fake_module = types.ModuleType("playwright.sync_api")
        fake_module.sync_playwright = lambda: _FakeSyncPlaywrightCM(chromium)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_module)

    class _FailAfterPeakBlob(_StreamingFakeBlob):
        def __init__(self, tempdir_capture: dict[str, Path], should_fail: bool) -> None:
            super().__init__()
            self._tempdir_capture = tempdir_capture
            self._should_fail = should_fail
            self.peak_disk_bytes: int | None = None

        def upload(self, sensitivity: str, path: str, data: Any, sha256: str) -> None:
            tempdir = self._tempdir_capture["path"]
            self.peak_disk_bytes = sum(f.stat().st_size for f in tempdir.rglob("*") if f.is_file())
            if self._should_fail:
                raise RuntimeError("simulated upload failure")
            super().upload(sensitivity, path, data, sha256)

    class _DiskFootprintExportPlan:
        name = "_disk_footprint_export_plan"

        def validate_config(self, config: dict[str, Any]) -> list[str]:
            del config
            return []

        def export(self, session: Any, partition_key: str, config: dict[str, Any], ctx: Any) -> Any:
            del partition_key, config, ctx
            yield session.click_and_await_download(role="button", name="Export")

    contract = IngestContract(
        source_id="77777777-8888-9999-aaaa-bbbbbbbbbbbb",
        source_name="browser-export-disk-acceptance",
        sensitivity="public",
        pattern="browser_export",
        prefix_template="browser_disk/{partition_key}",
        extract=(),
        strip_extensions=(),
        pattern_config={
            "export_plan": "_disk_footprint_export_plan",
            "export_config": {},
            "allowed_hosts": ["340bopais.hrsa.gov"],
            "partition": {"mode": "dynamic", "cadence": "daily", "anchor_tz": "UTC"},
        },
        compliance_review="SECURITY.md#disk-acceptance",
    )

    before_plans = dict(EXPORT_PLANS)
    register_export_plan(_DiskFootprintExportPlan())
    pattern = BrowserExportPattern()
    ctx = IngestContext(log=logging.getLogger("moncpipelib.test.browser_export_disk"))
    spec = PartitionSpec(key="2026-08-06", metadata={})

    try:
        # --- Success path ---
        tempdir_capture: dict[str, Path] = {}
        _install_fake_playwright(tempdir_capture)
        blob = _FailAfterPeakBlob(tempdir_capture, should_fail=False)
        with freeze_time("2026-08-06 12:00:00"):
            pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

        assert blob.peak_disk_bytes is not None
        assert blob.peak_disk_bytes <= _DISK_FOOTPRINT_MULTIPLE * member_size, (
            f"peak on-disk footprint was {blob.peak_disk_bytes / 1024 / 1024:.1f} MiB for a "
            f"{member_size / 1024 / 1024:.0f} MiB download -- exceeds the "
            f"{_DISK_FOOTPRINT_MULTIPLE}x bound."
        )
        assert not tempdir_capture["path"].exists()

        # --- Failure path ---
        tempdir_capture = {}
        _install_fake_playwright(tempdir_capture)
        blob = _FailAfterPeakBlob(tempdir_capture, should_fail=True)
        with (
            freeze_time("2026-08-06 12:00:00"),
            pytest.raises(RuntimeError, match="simulated upload failure"),
        ):
            pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

        assert blob.peak_disk_bytes is not None
        assert blob.peak_disk_bytes <= _DISK_FOOTPRINT_MULTIPLE * member_size
        assert not tempdir_capture["path"].exists()
    finally:
        EXPORT_PLANS.clear()
        EXPORT_PLANS.update(before_plans)
