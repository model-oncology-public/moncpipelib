"""Tests for HttpUrlsPattern.

httpx is mocked via respx; the blob resource is a tiny in-memory fake
that records uploads and serves sha256 from the last-recorded state.
This keeps the tests focused on the pattern's extract + hash-compare
logic without dragging in the Azure SDK.
"""

from __future__ import annotations

import hashlib
import io
import logging
import zipfile
from typing import IO, Literal

import httpx
import pytest
import respx

from moncpipelib.contracts.models import IngestContract
from moncpipelib.ingest.patterns.http_urls import HttpUrlsPattern
from moncpipelib.ingest.types import IngestContext, PartitionSpec


class FakeBlob:
    """In-memory stand-in for BlobStorageResource.

    Only implements the two methods HttpUrlsPattern actually calls:
    ``read_sha256_metadata`` and ``upload``. Mirrors the real contract --
    uploads set the sha256 that subsequent reads return.
    """

    def __init__(self, preloaded: dict[str, str] | None = None) -> None:
        # path -> (data, sha256)
        self.blobs: dict[str, tuple[bytes, str]] = {}
        if preloaded:
            for path, sha in preloaded.items():
                self.blobs[path] = (b"<existing>", sha)

    def read_sha256_metadata(self, sensitivity: str, path: str) -> str | None:
        del sensitivity
        entry = self.blobs.get(path)
        return entry[1] if entry else None

    def upload(
        self,
        sensitivity: str,
        path: str,
        data: bytes | IO[bytes],
        sha256: str,
    ) -> None:
        del sensitivity
        # Per #239 the pattern hands a file handle, not bytes.  Drain it
        # so existing equality assertions against ``self.blobs`` keep
        # working without each test having to know the difference.
        body = data if isinstance(data, bytes) else data.read()
        self.blobs[path] = (body, sha256)


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _contract(
    urls: list[str],
    partition_key: str = "2024-01-01",
    extract: tuple[str, ...] = ("zip",),
    strip_extensions: tuple[str, ...] = (),
    sensitivity: Literal["public", "confidential", "phi"] = "public",
    fetch_override: dict[str, object] | None = None,
    validate_content: dict[str, object] | None = None,
) -> IngestContract:
    fetch: dict[str, object] = {"retries": 0, "timeout_s": 5, "connect_timeout_s": 1}
    if fetch_override is not None:
        fetch.update(fetch_override)
    pattern_config: dict[str, object] = {
        "fetch": fetch,
        "periods": [{"partition_key": partition_key, "urls": urls}],
    }
    if validate_content is not None:
        pattern_config["validate_content"] = validate_content
    return IngestContract(
        source_id="11111111-1111-1111-1111-111111111111",
        source_name="cms-asp",
        sensitivity=sensitivity,
        pattern="http_urls",
        prefix_template="cms_asp/{partition_key}",
        extract=extract,
        strip_extensions=strip_extensions,
        pattern_config=pattern_config,
    )


def _spec(partition_key: str = "2024-01-01") -> PartitionSpec:
    return PartitionSpec(key=partition_key, metadata={"partition_key": partition_key})


def _ctx() -> IngestContext:
    # Real logger so caplog can capture pattern-emitted log lines.
    return IngestContext(log=logging.getLogger("moncpipelib.test.http_urls"))


@respx.mock
def test_happy_path_downloads_extracts_uploads() -> None:
    payload = _zip_bytes({"crosswalk.csv": b"col_a,col_b\n1,2\n"})
    respx.get("https://example.com/a.zip").respond(200, content=payload)

    contract = _contract(["https://example.com/a.zip"])
    blob = FakeBlob()

    results = HttpUrlsPattern().materialize_partition(contract, _spec(), blob, _ctx())  # type: ignore[arg-type]

    assert len(results) == 1
    [r] = results
    assert r.path == "cms_asp/2024-01-01/crosswalk.csv"
    assert r.action == "uploaded"
    assert r.sha256 == hashlib.sha256(b"col_a,col_b\n1,2\n").hexdigest()
    assert r.size_bytes == len(b"col_a,col_b\n1,2\n")
    assert "cms_asp/2024-01-01/crosswalk.csv" in blob.blobs


@respx.mock
def test_sha256_match_skips_upload() -> None:
    file_data = b"unchanged,row\n"
    payload = _zip_bytes({"crosswalk.csv": file_data})
    respx.get("https://example.com/a.zip").respond(200, content=payload)

    contract = _contract(["https://example.com/a.zip"])
    known_sha = hashlib.sha256(file_data).hexdigest()
    blob = FakeBlob(preloaded={"cms_asp/2024-01-01/crosswalk.csv": known_sha})

    results = HttpUrlsPattern().materialize_partition(contract, _spec(), blob, _ctx())  # type: ignore[arg-type]

    [r] = results
    assert r.action == "skipped"
    # blob content must not be replaced
    assert blob.blobs["cms_asp/2024-01-01/crosswalk.csv"][0] == b"<existing>"


@respx.mock
def test_sha256_mismatch_uploads() -> None:
    new_payload = b"changed,row\n"
    respx.get("https://example.com/a.zip").respond(
        200, content=_zip_bytes({"crosswalk.csv": new_payload})
    )

    contract = _contract(["https://example.com/a.zip"])
    blob = FakeBlob(preloaded={"cms_asp/2024-01-01/crosswalk.csv": "stale-sha"})

    results = HttpUrlsPattern().materialize_partition(contract, _spec(), blob, _ctx())  # type: ignore[arg-type]

    [r] = results
    assert r.action == "uploaded"
    assert blob.blobs["cms_asp/2024-01-01/crosswalk.csv"][0] == new_payload


@respx.mock
def test_strip_extensions_applied() -> None:
    # .xls on a file that's actually CSV content (CMS ASP's real shape):
    # after strip the key should be "payment_limit.csv".
    payload = _zip_bytes({"payment_limit.csv.xls": b"x,y\n"})
    respx.get("https://example.com/a.zip").respond(200, content=payload)

    contract = _contract(
        ["https://example.com/a.zip"],
        strip_extensions=(".xls", ".xlsx"),
    )
    blob = FakeBlob()

    results = HttpUrlsPattern().materialize_partition(contract, _spec(), blob, _ctx())  # type: ignore[arg-type]

    [r] = results
    assert r.path == "cms_asp/2024-01-01/payment_limit.csv"


@respx.mock
def test_multiple_urls_produce_union() -> None:
    respx.get("https://example.com/a.zip").respond(200, content=_zip_bytes({"a.csv": b"a1\n"}))
    respx.get("https://example.com/b.zip").respond(200, content=_zip_bytes({"b.csv": b"b1\n"}))

    contract = _contract(["https://example.com/a.zip", "https://example.com/b.zip"])
    blob = FakeBlob()

    results = HttpUrlsPattern().materialize_partition(contract, _spec(), blob, _ctx())  # type: ignore[arg-type]

    paths = {r.path for r in results}
    assert paths == {
        "cms_asp/2024-01-01/a.csv",
        "cms_asp/2024-01-01/b.csv",
    }


@respx.mock
def test_http_error_raises() -> None:
    respx.get("https://example.com/a.zip").respond(500, content=b"boom")

    contract = _contract(["https://example.com/a.zip"])

    with pytest.raises(httpx.HTTPStatusError):
        HttpUrlsPattern().materialize_partition(contract, _spec(), FakeBlob(), _ctx())  # type: ignore[arg-type]


def test_discover_partitions_enumerates_periods() -> None:
    contract = _contract(
        ["https://example.com/a.zip"],
        partition_key="2024-01-01",
    )
    specs = HttpUrlsPattern().discover_partitions(contract, _ctx())
    assert [s.key for s in specs] == ["2024-01-01"]
    assert specs[0].metadata["urls"] == ["https://example.com/a.zip"]


def test_missing_partition_key_raises() -> None:
    contract = _contract(["https://example.com/a.zip"], partition_key="2024-01-01")
    with pytest.raises(KeyError, match="no-such-partition"):
        HttpUrlsPattern().materialize_partition(
            contract,
            PartitionSpec(key="no-such-partition"),
            FakeBlob(),  # type: ignore[arg-type]
            _ctx(),
        )


# ---------------------------------------------------------------------------
# Streaming (Phase 2 refactor)
# ---------------------------------------------------------------------------


@respx.mock
def test_large_archive_does_not_load_into_memory() -> None:
    """The streaming refactor downloads to a tempfile and extracts via
    iterator.  Sanity-check: a multi-MB payload materializes without
    error -- the in-memory equality assertions in earlier tests would
    pass even on the broken bytes-path, so this test just exercises a
    larger size to catch obvious regressions.
    """
    one_megabyte = b"x" * (1 << 20)
    payload = _zip_bytes({"big.csv": one_megabyte})
    respx.get("https://example.com/big.zip").respond(200, content=payload)

    contract = _contract(["https://example.com/big.zip"])
    blob = FakeBlob()

    [result] = HttpUrlsPattern().materialize_partition(  # type: ignore[arg-type]
        contract, _spec(), blob, _ctx()
    )
    assert result.action == "uploaded"
    assert result.size_bytes == len(one_megabyte)


# ---------------------------------------------------------------------------
# Redirect handling (#211)
# ---------------------------------------------------------------------------


@respx.mock
def test_follows_redirect_by_default() -> None:
    # Models the observed CMS reissue: requested URL 301s to a re-stamped URL
    # that serves the actual zip. Pattern must transparently follow.
    payload = _zip_bytes({"crosswalk.csv": b"row\n"})
    respx.get("https://www.cms.gov/files/zip/april-2026-old.zip").respond(
        301,
        headers={"Location": "https://www.cms.gov/files/zip/april-2026-new.zip"},
    )
    respx.get("https://www.cms.gov/files/zip/april-2026-new.zip").respond(200, content=payload)

    contract = _contract(["https://www.cms.gov/files/zip/april-2026-old.zip"])
    blob = FakeBlob()

    results = HttpUrlsPattern().materialize_partition(contract, _spec(), blob, _ctx())  # type: ignore[arg-type]

    [r] = results
    assert r.action == "uploaded"
    assert r.path == "cms_asp/2024-01-01/crosswalk.csv"


@respx.mock
def test_follow_redirects_disabled_via_config() -> None:
    # Opt-out: a phi/confidential contract may want strict semantics.
    respx.get("https://www.cms.gov/files/zip/old.zip").respond(
        301,
        headers={"Location": "https://www.cms.gov/files/zip/new.zip"},
    )

    contract = _contract(
        ["https://www.cms.gov/files/zip/old.zip"],
        fetch_override={"follow_redirects": False},
    )

    with pytest.raises(httpx.HTTPStatusError):
        HttpUrlsPattern().materialize_partition(contract, _spec(), FakeBlob(), _ctx())  # type: ignore[arg-type]


@respx.mock
def test_user_agent_threaded_from_fetch_config() -> None:
    """``fetch.user_agent`` reaches the payload GET (#413)."""
    payload = _zip_bytes({"prices.csv": b"row\n"})
    route = respx.get("https://www.cms.gov/files/zip/data.zip").respond(200, content=payload)

    contract = _contract(
        ["https://www.cms.gov/files/zip/data.zip"],
        fetch_override={"user_agent": "ExampleOrgDataPlatform/1.0"},
    )
    HttpUrlsPattern().materialize_partition(contract, _spec(), FakeBlob(), _ctx())  # type: ignore[arg-type]

    assert route.calls.last.request.headers["User-Agent"] == "ExampleOrgDataPlatform/1.0"


@respx.mock
def test_default_user_agent_when_not_configured() -> None:
    """No ``fetch.user_agent`` -> httpx default UA on the wire (#413)."""
    payload = _zip_bytes({"prices.csv": b"row\n"})
    route = respx.get("https://www.cms.gov/files/zip/data.zip").respond(200, content=payload)

    contract = _contract(["https://www.cms.gov/files/zip/data.zip"])
    HttpUrlsPattern().materialize_partition(contract, _spec(), FakeBlob(), _ctx())  # type: ignore[arg-type]

    assert route.calls.last.request.headers["User-Agent"].startswith("python-httpx/")


@respx.mock
def test_user_agent_coerced_to_str_for_hand_rolled_contracts() -> None:
    """Contracts built directly in Python bypass the loader; a non-str
    ``user_agent`` is coerced like the neighboring knobs instead of
    exploding inside ``httpx.Client`` construction (TypeError)."""
    payload = _zip_bytes({"prices.csv": b"row\n"})
    route = respx.get("https://www.cms.gov/files/zip/data.zip").respond(200, content=payload)

    contract = _contract(
        ["https://www.cms.gov/files/zip/data.zip"],
        fetch_override={"user_agent": 123},
    )
    HttpUrlsPattern().materialize_partition(contract, _spec(), FakeBlob(), _ctx())  # type: ignore[arg-type]

    assert route.calls.last.request.headers["User-Agent"] == "123"


@respx.mock
def test_redirect_chain_logged_via_redacting_hooks(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The pattern itself no longer logs redirects directly.  Audit
    trail comes from the redacting client's event hooks: each request
    in the chain produces one ``method=GET host=... path=...`` line
    and each response produces a ``status=...`` line.  This test pins
    that both hops appear in the audit log."""
    payload = _zip_bytes({"crosswalk.csv": b"row\n"})
    respx.get("https://www.cms.gov/files/zip/old.zip").respond(
        301,
        headers={"Location": "https://www.cms.gov/files/zip/new.zip"},
    )
    respx.get("https://www.cms.gov/files/zip/new.zip").respond(200, content=payload)

    contract = _contract(["https://www.cms.gov/files/zip/old.zip"])

    with caplog.at_level("INFO", logger="moncpipelib.ingest._http"):
        HttpUrlsPattern().materialize_partition(contract, _spec(), FakeBlob(), _ctx())  # type: ignore[arg-type]

    log_lines = "\n".join(r.message for r in caplog.records)
    assert "/files/zip/old.zip" in log_lines  # original
    assert "/files/zip/new.zip" in log_lines  # post-redirect


# ---------------------------------------------------------------------------
# validate_content (per #228)
# ---------------------------------------------------------------------------


_HTML_BODY = b"<!DOCTYPE html>\n<html><body>not found</body></html>"
_HTML_BODY_LEADING_WS = b"\n  \r\n<!DOCTYPE html><html></html>"


def _captured_validate_content_log_text(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(r.message for r in caplog.records if "validate_content" in r.message)


@respx.mock
def test_validate_content_unset_preserves_union_semantics() -> None:
    """v0.27 behavior: every URL in the period contributes its files.

    Sanity check that the new code path (validate_content unset) matches
    the existing ``test_multiple_urls_produce_union`` semantics.
    """
    respx.get("https://example.com/a.zip").respond(200, content=_zip_bytes({"a.csv": b"a\n"}))
    respx.get("https://example.com/b.zip").respond(200, content=_zip_bytes({"b.csv": b"b\n"}))

    contract = _contract(["https://example.com/a.zip", "https://example.com/b.zip"])
    blob = FakeBlob()
    results = HttpUrlsPattern().materialize_partition(contract, _spec(), blob, _ctx())  # type: ignore[arg-type]

    assert {r.path for r in results} == {
        "cms_asp/2024-01-01/a.csv",
        "cms_asp/2024-01-01/b.csv",
    }


@respx.mock
def test_validate_content_first_url_passes_predicate() -> None:
    """Happy path: the first URL passes content_type_in; the second is
    never fetched (tested by leaving it unmocked + asserting
    `route.called` after)."""
    payload = _zip_bytes({"data.csv": b"row\n"})
    first = respx.get("https://example.com/april.zip").respond(
        200, content=payload, headers={"Content-Type": "application/zip"}
    )
    second = respx.get("https://example.com/march.zip").respond(200, content=payload)

    contract = _contract(
        ["https://example.com/april.zip", "https://example.com/march.zip"],
        validate_content={"content_type_in": ["application/zip"]},
    )
    blob = FakeBlob()
    results = HttpUrlsPattern().materialize_partition(contract, _spec(), blob, _ctx())  # type: ignore[arg-type]

    [r] = results
    assert r.path == "cms_asp/2024-01-01/data.csv"
    assert first.called
    assert not second.called  # fallback list: stop on first valid


@respx.mock
def test_validate_content_falls_back_when_first_returns_html(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """FDA Purplebook scenario: April returns 200+HTML, fall back to March."""
    respx.get("https://example.com/april.zip").respond(
        200, content=_HTML_BODY, headers={"Content-Type": "text/html; charset=utf-8"}
    )
    march_payload = _zip_bytes({"data.csv": b"row\n"})
    respx.get("https://example.com/march.zip").respond(
        200, content=march_payload, headers={"Content-Type": "application/zip"}
    )

    contract = _contract(
        ["https://example.com/april.zip", "https://example.com/march.zip"],
        validate_content={"content_type_in": ["application/zip"]},
    )
    blob = FakeBlob()

    with caplog.at_level("INFO"):
        results = HttpUrlsPattern().materialize_partition(contract, _spec(), blob, _ctx())  # type: ignore[arg-type]

    [r] = results
    assert r.path == "cms_asp/2024-01-01/data.csv"
    log_text = _captured_validate_content_log_text(caplog)
    assert "url_index=0" in log_text
    assert "reason=content_type" in log_text
    assert "text/html" in log_text


@respx.mock
def test_validate_content_first_bytes_match_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Server returns the right Content-Type but body is HTML.
    `reject_first_bytes_match` catches it."""
    respx.get("https://example.com/april.zip").respond(
        200,
        content=_HTML_BODY,
        headers={"Content-Type": "application/octet-stream"},
    )
    march_payload = _zip_bytes({"data.csv": b"row\n"})
    respx.get("https://example.com/march.zip").respond(
        200, content=march_payload, headers={"Content-Type": "application/octet-stream"}
    )

    contract = _contract(
        [
            "https://example.com/april.zip",
            "https://example.com/march.zip",
        ],
        validate_content={
            "content_type_in": ["application/octet-stream", "application/zip"],
            "reject_first_bytes_match": ["<!DOCTYPE", "<html"],
        },
    )
    blob = FakeBlob()
    with caplog.at_level("INFO"):
        results = HttpUrlsPattern().materialize_partition(contract, _spec(), blob, _ctx())  # type: ignore[arg-type]

    [r] = results
    assert r.path == "cms_asp/2024-01-01/data.csv"
    assert "reason=first_bytes" in _captured_validate_content_log_text(caplog)


@respx.mock
def test_validate_content_lstrip_before_first_bytes_check() -> None:
    """HTML often leads with whitespace (e.g. ``\\n<!DOCTYPE``).
    Predicate must lstrip before matching."""
    respx.get("https://example.com/april.zip").respond(
        200,
        content=_HTML_BODY_LEADING_WS,
        headers={"Content-Type": "application/octet-stream"},
    )
    payload = _zip_bytes({"data.csv": b"row\n"})
    respx.get("https://example.com/march.zip").respond(
        200, content=payload, headers={"Content-Type": "application/octet-stream"}
    )

    contract = _contract(
        ["https://example.com/april.zip", "https://example.com/march.zip"],
        validate_content={
            "content_type_in": ["application/octet-stream"],
            "reject_first_bytes_match": ["<!DOCTYPE"],
        },
    )
    blob = FakeBlob()
    [r] = HttpUrlsPattern().materialize_partition(contract, _spec(), blob, _ctx())  # type: ignore[arg-type]
    assert r.path == "cms_asp/2024-01-01/data.csv"


@respx.mock
def test_validate_content_all_urls_exhausted_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When every URL fails the predicate, raise IngestResolutionError
    with aggregated rejection context.  No raw URLs in the message."""
    from moncpipelib.ingest.exceptions import IngestResolutionError

    respx.get("https://example.com/april.zip?token=secret").respond(
        200, content=_HTML_BODY, headers={"Content-Type": "text/html"}
    )
    respx.get("https://example.com/march.zip?token=secret").respond(
        200, content=_HTML_BODY, headers={"Content-Type": "text/html"}
    )

    contract = _contract(
        [
            "https://example.com/april.zip?token=secret",
            "https://example.com/march.zip?token=secret",
        ],
        validate_content={"content_type_in": ["application/zip"]},
    )

    with caplog.at_level("INFO"), pytest.raises(IngestResolutionError) as excinfo:
        HttpUrlsPattern().materialize_partition(  # type: ignore[arg-type]
            contract, _spec(), FakeBlob(), _ctx()
        )

    error_text = str(excinfo.value)
    log_text = _captured_validate_content_log_text(caplog)

    # Aggregated context lists every rejection by index.
    assert "url_index=0" in error_text
    assert "url_index=1" in error_text

    # Defensive: query-string secret must not leak into the error or logs.
    assert "token=secret" not in error_text
    assert "token=secret" not in log_text
    assert "secret" not in error_text


@respx.mock
def test_validate_content_content_type_match_ignores_charset_suffix() -> None:
    """``application/zip`` matches ``application/zip; charset=utf-8``."""
    payload = _zip_bytes({"data.csv": b"row\n"})
    respx.get("https://example.com/a.zip").respond(
        200, content=payload, headers={"Content-Type": "application/zip; charset=utf-8"}
    )

    contract = _contract(
        ["https://example.com/a.zip"],
        validate_content={"content_type_in": ["application/zip"]},
    )
    [r] = HttpUrlsPattern().materialize_partition(  # type: ignore[arg-type]
        contract, _spec(), FakeBlob(), _ctx()
    )
    assert r.path == "cms_asp/2024-01-01/data.csv"


# ---------------------------------------------------------------------------
# partition_metadata (per #256)
# ---------------------------------------------------------------------------


def test_partition_metadata_returns_period_dict() -> None:
    """Per #256: ``HttpUrlsPattern.partition_metadata`` must return the
    period dict for the matching key, mirroring the shape that
    ``discover_partitions`` places on ``PartitionSpec.metadata``.

    Symmetric with :class:`ApiResolverPattern` so the dispatcher's
    manifest-write path is uniform across patterns -- and so a future
    ``http_urls`` consumer using ``FromIngestTemplate`` would just work.
    """
    contract = _contract(["https://example.com/a.zip"], partition_key="2024-01-01")

    fields = HttpUrlsPattern().partition_metadata(contract, "2024-01-01", _ctx())  # type: ignore[arg-type]

    assert fields["partition_key"] == "2024-01-01"
    assert fields["urls"] == ["https://example.com/a.zip"]


def test_partition_metadata_returns_empty_for_unknown_key() -> None:
    """A ``partition_key`` not present in any period yields ``{}``.  The
    dispatcher then falls back to ``partition_spec.metadata``, which
    keeps existing call sites working even against an out-of-band key."""
    contract = _contract(["https://example.com/a.zip"], partition_key="2024-01-01")

    fields = HttpUrlsPattern().partition_metadata(contract, "1999-01-01", _ctx())  # type: ignore[arg-type]

    assert fields == {}


# ---------------------------------------------------------------------------
# Non-archive payload filename precedence chain (#270)
# ---------------------------------------------------------------------------


def _non_archive_contract(
    urls: list[str],
    *,
    partition_key: str = "2024-01-01",
    payload_filename_template: str | None = None,
) -> IngestContract:
    """Helper: builds an http_urls contract with ``extract: ()``."""
    return IngestContract(
        source_id="11111111-1111-1111-1111-111111111111",
        source_name="cms-asp",
        sensitivity="public",
        pattern="http_urls",
        prefix_template="cms_asp/{partition_key}",
        extract=(),
        strip_extensions=(),
        pattern_config={
            "fetch": {"retries": 0, "timeout_s": 5, "connect_timeout_s": 1},
            "periods": [{"partition_key": partition_key, "urls": urls}],
        },
        payload_filename_template=payload_filename_template,
    )


@respx.mock
def test_non_archive_url_basename_used_as_filename() -> None:
    """Default fallback for ``extract: []``: the sanitized URL basename
    becomes the landed filename, replacing the helper's ``__payload__``
    sentinel.  Regression guard for the SEER CPC SMVL motivating bug."""
    body = b"col_a,col_b\n1,2\n"
    respx.get("https://example.com/V2024B_V2025B_V2026A_CPC_SMVL.csv").respond(200, content=body)

    contract = _non_archive_contract(["https://example.com/V2024B_V2025B_V2026A_CPC_SMVL.csv"])
    blob = FakeBlob()

    results = HttpUrlsPattern().materialize_partition(contract, _spec(), blob, _ctx())  # type: ignore[arg-type]

    [r] = results
    assert r.path == "cms_asp/2024-01-01/V2024B_V2025B_V2026A_CPC_SMVL.csv"
    assert r.action == "uploaded"


@respx.mock
def test_non_archive_url_with_percent_escape_is_sanitized() -> None:
    """URL basenames with percent-escapes are URL-decoded; spaces
    collapse to ``_`` so the landed name stays parseable."""
    body = b"hello"
    respx.get("https://example.com/odd%20file.csv").respond(200, content=body)

    contract = _non_archive_contract(["https://example.com/odd%20file.csv"])
    blob = FakeBlob()

    results = HttpUrlsPattern().materialize_partition(contract, _spec(), blob, _ctx())  # type: ignore[arg-type]

    [r] = results
    assert r.path == "cms_asp/2024-01-01/odd_file.csv"


@respx.mock
def test_non_archive_url_with_no_basename_raises() -> None:
    """When the URL has no path component AND no template / hint /
    Content-Disposition is set, materialization fails loudly so the
    contract author surfaces the gap."""
    from moncpipelib.ingest.exceptions import IngestResolutionError

    respx.get("https://api.example/").respond(200, content=b"x")

    contract = _non_archive_contract(["https://api.example/"])
    blob = FakeBlob()

    with pytest.raises(IngestResolutionError, match="payload_filename_template"):
        HttpUrlsPattern().materialize_partition(contract, _spec(), blob, _ctx())  # type: ignore[arg-type]


@respx.mock
def test_non_archive_template_wins_over_url_basename() -> None:
    """A ``payload_filename_template`` overrides the URL basename even
    when the URL has a perfectly good name -- the contract author has
    expressed an explicit preference."""
    respx.get("https://example.com/upstream.csv").respond(200, content=b"x")

    contract = _non_archive_contract(
        ["https://example.com/upstream.csv"],
        payload_filename_template="{source_name}_{partition_key}.csv",
    )
    blob = FakeBlob()

    results = HttpUrlsPattern().materialize_partition(contract, _spec(), blob, _ctx())  # type: ignore[arg-type]

    [r] = results
    assert r.path == "cms_asp/2024-01-01/cms-asp_2024-01-01.csv"


@respx.mock
def test_non_archive_content_disposition_wins_over_url_basename() -> None:
    """When the server supplies ``Content-Disposition: filename=...``,
    the parsed name is sanitized and used over the URL basename."""
    respx.get("https://api.example/download").respond(
        200,
        content=b"x",
        headers={"Content-Disposition": 'attachment; filename="report-2024.csv"'},
    )

    contract = _non_archive_contract(["https://api.example/download"])
    blob = FakeBlob()

    results = HttpUrlsPattern().materialize_partition(contract, _spec(), blob, _ctx())  # type: ignore[arg-type]

    [r] = results
    assert r.path == "cms_asp/2024-01-01/report-2024.csv"


@respx.mock
def test_archive_extract_unaffected_by_precedence_chain() -> None:
    """Regression guard: ``extract: [zip]`` is unchanged.  Archive
    members keep their in-archive paths; the precedence chain runs only
    for the helper-emitted ``__payload__`` sentinel, which the archive
    branch never produces."""
    payload = _zip_bytes({"crosswalk.csv": b"col_a,col_b\n1,2\n"})
    respx.get("https://example.com/a.zip").respond(200, content=payload)

    contract = _contract(
        ["https://example.com/a.zip"],
        # Setting a template should NOT affect archive contracts.
    )
    # template would be ignored for archive contracts; build by mutating
    # _contract output to keep the helper signature small.
    contract = IngestContract(
        source_id=contract.source_id,
        source_name=contract.source_name,
        sensitivity=contract.sensitivity,
        pattern=contract.pattern,
        prefix_template=contract.prefix_template,
        extract=("zip",),
        strip_extensions=contract.strip_extensions,
        pattern_config=contract.pattern_config,
        extract_filter=contract.extract_filter,
        payload_filename_template="OVERRIDE.csv",
    )
    blob = FakeBlob()

    results = HttpUrlsPattern().materialize_partition(contract, _spec(), blob, _ctx())  # type: ignore[arg-type]

    [r] = results
    # The in-archive name wins, NOT the template.
    assert r.path == "cms_asp/2024-01-01/crosswalk.csv"


@respx.mock
def test_non_archive_emits_decision_log(caplog: pytest.LogCaptureFixture) -> None:
    """A single INFO line records which precedence level produced the
    name -- searchable from logs alone (no shell access to landing pods)."""
    respx.get("https://example.com/data.csv").respond(200, content=b"x")

    contract = _non_archive_contract(["https://example.com/data.csv"])
    blob = FakeBlob()

    with caplog.at_level(logging.INFO, logger="moncpipelib.test.http_urls"):
        HttpUrlsPattern().materialize_partition(contract, _spec(), blob, _ctx())  # type: ignore[arg-type]

    captured = "\n".join(record.getMessage() for record in caplog.records)
    assert "ingest.payload_filename" in captured
    assert "source_level=url_basename" in captured
    assert "data.csv" in captured
