"""Tests for ``BrowserExportPattern`` (#463 Step 3).

No real browser: every test drives the pattern against a monkeypatched
``browser_session`` and a stub :class:`ExportPlan`. Coverage per the
executor briefs:

- discovery: single current-boundary partition, no browser touched,
  publication lag, metadata symmetry with discovery
- materialize happy path: upload + manifest, re-run skip
- D7's four failure conditions (stale partition, zero bytes, content
  rejected, control-not-found / timeout from the session) -- each leaves
  no manifest written
- D8's untrusted-filename handling: local-disk traversal already guarded
  by the session, blob-path traversal guarded by sanitization, the
  reserved manifest filename refused on both naming-chain branches, an
  unsanitizable name raising with an actionable message, and duplicate
  resolved filenames raising rather than silently overwriting
- the streaming primitives: ``_hash_file``'s one-pass hash+size, and the
  drift guard pinning browser_export's ``max_first_bytes_check`` default
  against ``http_urls``'
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import pytest
from freezegun import freeze_time

from moncpipelib.contracts.models import IngestContract
from moncpipelib.ingest import _browser
from moncpipelib.ingest._browser import _ERR_CONTROL_NOT_FOUND, _ERR_TIMEOUT
from moncpipelib.ingest.dispatcher import _MANIFEST_FILENAME, materialize_with_manifest
from moncpipelib.ingest.exceptions import IngestResolutionError
from moncpipelib.ingest.export_plans import EXPORT_PLANS, ExportedFile
from moncpipelib.ingest.patterns import browser_export, http_urls
from moncpipelib.ingest.patterns.browser_export import (
    _ERR_CONTENT_REJECTED,
    _ERR_DUPLICATE_FILENAME,
    _ERR_NO_FILENAME,
    _ERR_RESERVED_FILENAME,
    _ERR_SENSITIVITY_NOT_PUBLIC,
    _ERR_STALE_PARTITION,
    _ERR_ZERO_BYTES,
    BrowserExportPattern,
    _hash_file,
)
from moncpipelib.ingest.types import IngestContext, PartitionSpec

_UUID = "77777777-7777-7777-7777-777777777777"

# 2026-08-06 12:00 UTC is 08:00 EDT (America/New_York) -- same calendar day,
# no cross-midnight edge case, so the daily boundary is simply "2026-08-06".
_FREEZE_DAILY = "2026-08-06 12:00:00"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_export_plans() -> Iterator[None]:
    before = dict(EXPORT_PLANS)
    try:
        yield
    finally:
        EXPORT_PLANS.clear()
        EXPORT_PLANS.update(before)


class _StubExportPlan:
    """Duck-typed :class:`ExportPlan` driven by a per-test generator function."""

    def __init__(
        self,
        export_fn: Any,
        *,
        name: str = "_stub_export_plan",
        multi_file: bool = False,
    ) -> None:
        self.name = name
        self.multi_file = multi_file
        self._export_fn = export_fn

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        del config
        return []

    def export(
        self,
        session: Any,
        partition_key: str,
        config: dict[str, Any],
        ctx: IngestContext,
    ) -> Iterator[ExportedFile]:
        yield from self._export_fn(session, partition_key, config, ctx)


def _register(plan: Any) -> None:
    from moncpipelib.ingest.export_plans import register_export_plan

    register_export_plan(plan)


class _FakeSession:
    """Fake :class:`BrowserSession` for plans that call session methods.

    ``click_results`` is a queue: each element is either an
    :class:`ExportedFile` (returned) or an :class:`Exception` (raised).

    ``contains_raises`` fakes :meth:`BrowserSession.require_contains`
    (#464/#468 finding 12): the real containment logic is pinned against
    the real class in ``tests/test_ingest_browser_session.py``; this fake
    only needs to prove the *pattern* calls it and propagates its raise.
    """

    def __init__(
        self, click_results: list[Any] | None = None, *, contains_raises: bool = False
    ) -> None:
        self._click_results = list(click_results or [])
        self._contains_raises = contains_raises

    def click_and_await_download(self, **kwargs: Any) -> ExportedFile:
        del kwargs
        result = self._click_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result  # type: ignore[no-any-return]

    def require_contains(self, path: Path) -> None:
        if self._contains_raises:
            raise IngestResolutionError(
                f"browser_export: exported file path {str(path)!r} is outside "
                "the session's own download directory -- refusing to hash or "
                "upload it"
            )


def _patch_browser_session(monkeypatch: pytest.MonkeyPatch, session: Any = None) -> None:
    @contextmanager
    def _fake_browser_session(**kwargs: Any) -> Iterator[Any]:
        del kwargs
        yield session if session is not None else _FakeSession()

    monkeypatch.setattr(browser_export, "browser_session", _fake_browser_session)


def _patch_browser_session_capturing(
    monkeypatch: pytest.MonkeyPatch, session: Any = None
) -> dict[str, Any]:
    """Like :func:`_patch_browser_session`, but returns a dict populated with
    the kwargs ``browser_session(...)`` was actually called with.

    #464/#468 finding 2: the plain fake's ``del kwargs`` discarded
    everything, so no test ever asserted the contract's ``allowed_hosts``
    (or anything else) actually reached ``browser_session``.
    """
    captured: dict[str, Any] = {}

    @contextmanager
    def _fake_browser_session(**kwargs: Any) -> Iterator[Any]:
        captured.update(kwargs)
        yield session if session is not None else _FakeSession()

    monkeypatch.setattr(browser_export, "browser_session", _fake_browser_session)
    return captured


def _pattern_config(**overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "export_plan": "_stub_export_plan",
        "export_config": {},
        "allowed_hosts": ["340bopais.hrsa.gov"],
        "partition": {
            "mode": "dynamic",
            "cadence": "daily",
            "anchor_tz": "America/New_York",
        },
    }
    cfg.update(overrides)
    return cfg


def _contract(*, pattern_config: dict[str, Any] | None = None, **overrides: Any) -> IngestContract:
    kwargs: dict[str, Any] = {
        "source_id": _UUID,
        "source_name": "browser-export-test",
        "sensitivity": "public",
        "pattern": "browser_export",
        "prefix_template": "besrc/{partition_key}",
        "extract": (),
        "strip_extensions": (),
        "pattern_config": pattern_config if pattern_config is not None else _pattern_config(),
        "compliance_review": "SECURITY.md#stub",
    }
    kwargs.update(overrides)
    return IngestContract(**kwargs)


def _ctx() -> IngestContext:
    return IngestContext(log=logging.getLogger("moncpipelib.test.browser_export"))


class _InMemoryBlob:
    def __init__(self) -> None:
        self.store: dict[str, tuple[bytes, str]] = {}
        self.upload_calls: list[str] = []

    def read_sha256_metadata(self, sensitivity: str, path: str) -> str | None:
        del sensitivity
        entry = self.store.get(path)
        return entry[1] if entry else None

    def upload(self, sensitivity: str, path: str, data: bytes | IO[bytes], sha256: str) -> None:
        del sensitivity
        self.upload_calls.append(path)
        body = data if isinstance(data, bytes) else data.read()
        self.store[path] = (body, sha256)


def _write_and_export_one(tmp_path: Path, content: bytes, suggested_filename: str) -> Any:
    """Build an ``export_fn`` yielding one :class:`ExportedFile` over ``content``."""
    payload_path = tmp_path / "0000.download"
    payload_path.write_bytes(content)

    def _export(
        session: Any, partition_key: str, config: dict[str, Any], ctx: IngestContext
    ) -> Iterator[ExportedFile]:
        del session, partition_key, config, ctx
        yield ExportedFile(path=payload_path, suggested_filename=suggested_filename)

    return _export


# ---------------------------------------------------------------------------
# Discovery + partition key
# ---------------------------------------------------------------------------


def test_discover_partitions_returns_single_current_boundary() -> None:
    contract = _contract()
    _register(_StubExportPlan(lambda *_a: iter(())))
    pattern = BrowserExportPattern()
    with freeze_time(_FREEZE_DAILY):
        specs = pattern.discover_partitions(contract, _ctx())
    assert len(specs) == 1
    assert specs[0].key == "2026-08-06"


def test_discover_partitions_requires_no_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard for the mis-sited probe (D2): discovery must never
    touch the browser -- a process that only discovers partitions must
    need no browser installed."""

    def _raise_if_called(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("discover_partitions must not touch the browser")

    monkeypatch.setattr(browser_export, "browser_session", _raise_if_called)
    monkeypatch.setattr(_browser, "ensure_chromium_available", _raise_if_called)

    contract = _contract()
    pattern = BrowserExportPattern()
    with freeze_time(_FREEZE_DAILY):
        specs = pattern.discover_partitions(contract, _ctx())
    assert specs[0].key == "2026-08-06"


def test_discover_partitions_applies_publication_lag() -> None:
    """01:00 Eastern with a 3-hour lag yields the previous day's key."""
    contract = _contract(
        pattern_config=_pattern_config(
            partition={
                "mode": "dynamic",
                "cadence": "daily",
                "anchor_tz": "America/New_York",
                "publication_lag_hours": 3,
            }
        )
    )
    pattern = BrowserExportPattern()
    # 05:00 UTC == 01:00 EDT on 2026-08-06.
    with freeze_time("2026-08-06 05:00:00"):
        [spec] = pattern.discover_partitions(contract, _ctx())
    assert spec.key == "2026-08-05"


def test_partition_metadata_echoes_passed_key_not_the_clock() -> None:
    contract = _contract()
    pattern = BrowserExportPattern()
    with freeze_time(_FREEZE_DAILY):
        metadata = pattern.partition_metadata(contract, "2020-01-01", _ctx())
    assert metadata["snapshot_date"] == "2020-01-01"


def test_partition_metadata_emits_the_field_from_ingest_reads() -> None:
    import datetime as _dt

    contract = _contract()
    pattern = BrowserExportPattern()
    with freeze_time(_FREEZE_DAILY):
        metadata = pattern.partition_metadata(contract, "2026-08-06", _ctx())
    assert "snapshot_date" in metadata
    assert _dt.date.fromisoformat(metadata["snapshot_date"]) == _dt.date(2026, 8, 6)


def test_discovery_and_metadata_agree() -> None:
    contract = _contract()
    pattern = BrowserExportPattern()
    ctx = _ctx()
    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        metadata = pattern.partition_metadata(contract, spec.key, ctx)
    assert spec.metadata == metadata


# ---------------------------------------------------------------------------
# Materialize -- happy path
# ---------------------------------------------------------------------------


def test_materialize_uploads_single_file_and_writes_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _contract()
    export_fn = _write_and_export_one(tmp_path, b"hello-export", "report.csv")
    _register(_StubExportPlan(export_fn))
    _patch_browser_session(monkeypatch)

    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        results = materialize_with_manifest(pattern, contract, spec, blob, ctx)

    assert len(results) == 1
    assert results[0].action == "uploaded"
    assert results[0].path == "besrc/2026-08-06/report.csv"

    from io import BytesIO

    from moncpipelib.ingest.manifest import IngestManifest

    manifest_bytes = blob.store["besrc/2026-08-06/_manifest.json"][0]
    manifest = IngestManifest.read_from(BytesIO(manifest_bytes))
    assert manifest.fields["snapshot_date"] == "2026-08-06"
    assert manifest.resolver["name"] == "_stub_export_plan"


def test_materialize_rerun_skips_on_identical_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _contract()
    export_fn = _write_and_export_one(tmp_path, b"stable-bytes", "report.csv")
    _register(_StubExportPlan(export_fn))
    _patch_browser_session(monkeypatch)

    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        materialize_with_manifest(pattern, contract, spec, blob, ctx)
        second = materialize_with_manifest(pattern, contract, spec, blob, ctx)

    assert second[0].action == "skipped"


# ---------------------------------------------------------------------------
# Materialize -- sensitivity backstop (pre-merge review gate finding 3)
# ---------------------------------------------------------------------------


def test_materialize_rejects_non_public_sensitivity_before_opening_a_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding 3: ``validate_ingest_contract_schema`` enforces
    ``sensitivity: public`` for ``browser_export`` at contract LOAD time,
    but ``IngestContract`` is a plain dataclass and ``materialize_partition``
    does not re-validate -- any caller constructing one directly (bypassing
    the loader) skips that check entirely and could drive a headless
    browser against a confidential/PHI source. This is the same class of
    gap as a mode-scoped write flag arriving via a kwarg and skipping the
    loader's static check -- it needs a materialize-time backstop.

    Asserts the backstop runs as an early action, before even opening a
    browser session: ``browser_session`` is monkeypatched to raise if
    called at all, so this test fails loudly (an ``AssertionError``, not
    the expected ``IngestResolutionError``) if the ordering regresses."""

    def _raise_if_browser_session_opened(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(
            "must not open a browser session for a non-public-sensitivity contract"
        )

    monkeypatch.setattr(browser_export, "browser_session", _raise_if_browser_session_opened)

    contract = _contract(sensitivity="confidential")
    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        with pytest.raises(IngestResolutionError) as exc_info:
            pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

    message = str(exc_info.value)
    assert _ERR_SENSITIVITY_NOT_PUBLIC in message
    assert "confidential" in message
    assert blob.upload_calls == []


# ---------------------------------------------------------------------------
# Materialize -- allowed_hosts / fetch.user_agent backstop (#464/#468 round 3)
# ---------------------------------------------------------------------------


def test_materialize_accepts_reserved_or_ip_shaped_allowed_hosts_entry_and_reaches_browser_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round 6 pin (docs/migrations/20260807_463-egress-allowlist-reframe.md):
    the backstop screens shape only. A hand-built contract's
    ``allowed_hosts: ["localhost"]`` -- rejected by the deny-list before
    round 6 -- now passes the backstop and reaches ``browser_session``.
    ``browser_session`` is monkeypatched to raise a unique sentinel
    exception rather than ``IngestResolutionError``, proving the backstop
    did not intercept the call."""

    class _SentinelBrowserSessionOpened(Exception):
        pass

    def _raise_sentinel(*_args: Any, **_kwargs: Any) -> Any:
        raise _SentinelBrowserSessionOpened

    monkeypatch.setattr(browser_export, "browser_session", _raise_sentinel)
    _register(_StubExportPlan(lambda *_a: iter(())))

    contract = _contract(pattern_config=_pattern_config(allowed_hosts=["localhost"]))
    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        with pytest.raises(_SentinelBrowserSessionOpened):
            pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

    assert blob.upload_calls == []


def test_materialize_accepts_ordinary_allowed_hosts_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for the backstop above: an ordinary, contract-load
    valid ``allowed_hosts`` list must not be rejected at materialize
    time."""
    captured = _patch_browser_session_capturing(monkeypatch)
    contract = _contract(pattern_config=_pattern_config(allowed_hosts=["340bopais.hrsa.gov"]))
    _register(_StubExportPlan(lambda *_a: iter(())))
    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

    assert captured["allowed_hosts"] == ["340bopais.hrsa.gov"]


def test_materialize_rejects_non_list_allowed_hosts_cfg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#463 egress-allowlist reframe, Step 1b: a hand-built contract could
    set ``allowed_hosts`` to a bare string -- iterating it would silently
    walk character by character instead of raising, and a non-iterable
    value (e.g. an int) would raise a raw ``TypeError`` rather than
    ``IngestResolutionError``. Guarded explicitly before the per-entry
    loop runs at all."""

    def _raise_if_browser_session_opened(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("must not open a browser session with a malformed allowed_hosts")

    monkeypatch.setattr(browser_export, "browser_session", _raise_if_browser_session_opened)

    contract = _contract(pattern_config=_pattern_config(allowed_hosts="340bopais.hrsa.gov"))
    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        with pytest.raises(IngestResolutionError) as exc_info:
            pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

    assert not isinstance(exc_info.value, TypeError)
    assert "list" in str(exc_info.value)
    assert blob.upload_calls == []


def test_materialize_rejects_empty_allowed_hosts_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#464/#468 round 4, J6: the loader already rejects an empty
    ``allowed_hosts`` list at contract load, but the backstop's own
    hand-rolled type guard checked only ``isinstance(allowed_hosts_cfg, list)``
    -- true of ``[]`` -- so a hand-built contract with an empty list ran
    the per-entry loop zero times and reached ``browser_session`` with a
    runtime allowlist matching nothing. That fails CLOSED (every request
    is aborted), but surfaces as a 60s navigation timeout naming the
    wrong cause rather than this diagnostic."""

    def _raise_if_browser_session_opened(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("must not open a browser session with an empty allowed_hosts list")

    monkeypatch.setattr(browser_export, "browser_session", _raise_if_browser_session_opened)

    contract = _contract(pattern_config=_pattern_config(allowed_hosts=[]))
    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        with pytest.raises(IngestResolutionError) as exc_info:
            pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

    assert "non-empty" in str(exc_info.value)
    assert blob.upload_calls == []


@pytest.mark.parametrize("entry", ["evil.com:8080", "*.hrsa.gov"])
def test_materialize_rejects_malformed_shape_allowed_hosts_entry(
    monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    """#463 egress-allowlist reframe, Step 1b: an entry with a port suffix
    or a leading wildcard label is rejected by the shape pre-screen the
    loader and this backstop both run -- the ONLY check either performs
    (round 6 removed the deny-list check that used to run alongside it)."""

    def _raise_if_browser_session_opened(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("must not open a browser session with a malformed allowed_hosts entry")

    monkeypatch.setattr(browser_export, "browser_session", _raise_if_browser_session_opened)

    contract = _contract(pattern_config=_pattern_config(allowed_hosts=[entry]))
    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        with pytest.raises(IngestResolutionError) as exc_info:
            pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

    assert entry in str(exc_info.value)
    assert "bare hostname" in str(exc_info.value)
    assert blob.upload_calls == []


def test_materialize_rejects_non_ascii_fetch_user_agent_before_opening_a_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round 3: ``validate_ingest_contract_schema`` requires a present
    ``fetch.user_agent`` to be non-empty printable ASCII (httpx encodes
    header values as strict ASCII), but a hand-built contract skips that
    check the identical way it skips the ``allowed_hosts`` one above --
    e.g. a control character or CRLF sequence that would reach an HTTP
    header raw."""

    def _raise_if_browser_session_opened(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("must not open a browser session with an invalid user_agent")

    monkeypatch.setattr(browser_export, "browser_session", _raise_if_browser_session_opened)

    contract = _contract(
        pattern_config=_pattern_config(fetch={"user_agent": "evil\r\nX-Injected: 1"})
    )
    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        with pytest.raises(IngestResolutionError) as exc_info:
            pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

    message = str(exc_info.value)
    assert "printable-ASCII" in message
    assert blob.upload_calls == []


def test_materialize_rejects_empty_fetch_user_agent_before_opening_a_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same backstop, empty-string half: mirrors the loader's own
    ``must be a non-empty string`` rejection."""

    def _raise_if_browser_session_opened(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("must not open a browser session with an invalid user_agent")

    monkeypatch.setattr(browser_export, "browser_session", _raise_if_browser_session_opened)

    contract = _contract(pattern_config=_pattern_config(fetch={"user_agent": "   "}))
    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        with pytest.raises(IngestResolutionError):
            pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

    assert blob.upload_calls == []


def test_materialize_does_not_require_fetch_user_agent_to_be_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The backstop re-checks FORMAT only, not presence: an absent
    ``fetch.user_agent`` degrades to chromium's default UA (a governance
    gap enforced at contract load, not a materialize-time security hole),
    so ``materialize_partition`` must not start rejecting every contract
    fixture that omits ``fetch`` entirely -- ``_pattern_config()``'s
    default has no ``fetch`` block at all."""
    captured = _patch_browser_session_capturing(monkeypatch)
    contract = _contract()
    _register(_StubExportPlan(lambda *_a: iter(())))
    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

    assert captured["user_agent"] is None


# ---------------------------------------------------------------------------
# Materialize -- D7's four failure conditions
# ---------------------------------------------------------------------------


def test_stale_partition_key_refused() -> None:
    contract = _contract()
    pattern = BrowserExportPattern()
    ctx = _ctx()
    stale_spec = PartitionSpec(key="2026-08-01", metadata={})
    blob = _InMemoryBlob()

    with (
        freeze_time("2026-08-10 12:00:00"),
        pytest.raises(IngestResolutionError) as exc_info,
    ):
        materialize_with_manifest(pattern, contract, stale_spec, blob, ctx)

    message = str(exc_info.value)
    assert _ERR_STALE_PARTITION in message
    assert "2026-08-01" in message
    assert "2026-08-10" in message
    assert not any(p.endswith(f"/{_MANIFEST_FILENAME}") for p in blob.store)


def test_stale_partition_guard_rejects_a_retry_crossing_the_midnight_rollover() -> None:
    """Pins D4a.1's accepted tradeoff (finding 9), not a bug: a key
    discovered at 23:55 in ``anchor_tz`` and retried 10 minutes later, at
    00:05 the next day, is refused as stale even though it is the SAME
    logical run and the wall-clock gap is minutes, not days. See
    ``materialize_partition``'s docstring for why this is deliberate."""
    contract = _contract()
    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    # 2026-08-06 23:55 America/New_York (EDT, UTC-4) == 2026-08-07 03:55 UTC.
    with freeze_time("2026-08-07 03:55:00"):
        [spec] = pattern.discover_partitions(contract, ctx)
    assert spec.key == "2026-08-06"

    # 10 minutes later, past local midnight: 2026-08-07 00:05 America/New_York
    # == 2026-08-07 04:05 UTC. A Dagster retry of the SAME run would present
    # this same spec.
    with (
        freeze_time("2026-08-07 04:05:00"),
        pytest.raises(IngestResolutionError, match=_ERR_STALE_PARTITION) as exc_info,
    ):
        materialize_with_manifest(pattern, contract, spec, blob, ctx)

    message = str(exc_info.value)
    assert "2026-08-06" in message
    assert "2026-08-07" in message


def test_zero_byte_download_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    contract = _contract()
    export_fn = _write_and_export_one(tmp_path, b"", "empty.csv")
    _register(_StubExportPlan(export_fn))
    _patch_browser_session(monkeypatch)

    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        with pytest.raises(IngestResolutionError, match=_ERR_ZERO_BYTES):
            materialize_with_manifest(pattern, contract, spec, blob, ctx)

    assert not any(p.endswith(f"/{_MANIFEST_FILENAME}") for p in blob.store)


def test_html_error_page_rejected_before_upload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _contract(
        pattern_config=_pattern_config(
            validate_content={"reject_first_bytes_match": ["<!DOCTYPE html", "<html"]}
        )
    )
    export_fn = _write_and_export_one(tmp_path, b"\n  <!DOCTYPE html>\n<html></html>", "error.csv")
    _register(_StubExportPlan(export_fn))
    _patch_browser_session(monkeypatch)

    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        with pytest.raises(IngestResolutionError, match=_ERR_CONTENT_REJECTED):
            materialize_with_manifest(pattern, contract, spec, blob, ctx)

    assert blob.upload_calls == []
    assert not any(p.endswith(f"/{_MANIFEST_FILENAME}") for p in blob.store)


def test_control_not_found_raises_from_session(monkeypatch: pytest.MonkeyPatch) -> None:
    error = IngestResolutionError(f"{_ERR_CONTROL_NOT_FOUND} (role='button' name='Export')")

    def _export(
        session: Any, partition_key: str, config: dict[str, Any], ctx: IngestContext
    ) -> Iterator[ExportedFile]:
        del partition_key, config, ctx
        yield session.click_and_await_download(role="button", name="Export")

    contract = _contract()
    _register(_StubExportPlan(_export))
    _patch_browser_session(monkeypatch, session=_FakeSession(click_results=[error]))

    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        with pytest.raises(IngestResolutionError) as exc_info:
            materialize_with_manifest(pattern, contract, spec, blob, ctx)

    message = str(exc_info.value)
    assert _ERR_CONTROL_NOT_FOUND in message
    assert "role='button'" in message
    assert "name='Export'" in message
    assert not any(p.endswith(f"/{_MANIFEST_FILENAME}") for p in blob.store)


def test_timeout_raises_from_session(monkeypatch: pytest.MonkeyPatch) -> None:
    error = IngestResolutionError(f"{_ERR_TIMEOUT}: no download completed")

    def _export(
        session: Any, partition_key: str, config: dict[str, Any], ctx: IngestContext
    ) -> Iterator[ExportedFile]:
        del partition_key, config, ctx
        yield session.click_and_await_download(role="button", name="Export")

    contract = _contract()
    _register(_StubExportPlan(_export))
    _patch_browser_session(monkeypatch, session=_FakeSession(click_results=[error]))

    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        with pytest.raises(IngestResolutionError, match=_ERR_TIMEOUT):
            materialize_with_manifest(pattern, contract, spec, blob, ctx)

    assert not any(p.endswith(f"/{_MANIFEST_FILENAME}") for p in blob.store)


def test_four_failure_tags_are_pairwise_distinguishable() -> None:
    tags = [_ERR_CONTROL_NOT_FOUND, _ERR_ZERO_BYTES, _ERR_TIMEOUT, _ERR_CONTENT_REJECTED]
    assert len(set(tags)) == 4
    assert all(not a.startswith(b) for a in tags for b in tags if a is not b)


@pytest.mark.parametrize("case", ["zero_bytes", "content_rejected", "control_not_found", "timeout"])
def test_no_manifest_written_on_any_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    pattern_config_overrides: dict[str, Any] = {}
    session: Any = None
    export_fn: Any

    if case == "zero_bytes":
        export_fn = _write_and_export_one(tmp_path, b"", "empty.csv")
    elif case == "content_rejected":
        pattern_config_overrides["validate_content"] = {
            "reject_first_bytes_match": ["<!DOCTYPE html", "<html"]
        }
        export_fn = _write_and_export_one(
            tmp_path, b"\n  <!DOCTYPE html>\n<html></html>", "error.csv"
        )
    elif case == "control_not_found":
        error = IngestResolutionError(f"{_ERR_CONTROL_NOT_FOUND} (role='button' name='Export')")
        session = _FakeSession(click_results=[error])

        def export_fn(
            s: Any, partition_key: str, config: dict[str, Any], ctx: IngestContext
        ) -> Iterator[ExportedFile]:
            del partition_key, config, ctx
            yield s.click_and_await_download(role="button", name="Export")
    else:
        error = IngestResolutionError(f"{_ERR_TIMEOUT}: no download completed")
        session = _FakeSession(click_results=[error])

        def export_fn(
            s: Any, partition_key: str, config: dict[str, Any], ctx: IngestContext
        ) -> Iterator[ExportedFile]:
            del partition_key, config, ctx
            yield s.click_and_await_download(role="button", name="Export")

    contract = _contract(pattern_config=_pattern_config(**pattern_config_overrides))
    _register(_StubExportPlan(export_fn))
    _patch_browser_session(monkeypatch, session=session)

    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        with pytest.raises(IngestResolutionError):
            materialize_with_manifest(pattern, contract, spec, blob, ctx)

    assert not any(p.endswith(f"/{_MANIFEST_FILENAME}") for p in blob.store)


# ---------------------------------------------------------------------------
# D8 -- untrusted filenames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("suggested_filename", "unsafe"),
    [
        ("../../other-source/x.json", False),
        ("..", True),
        (".", True),
    ],
)
def test_traversal_suggested_filename_cannot_escape_partition_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, suggested_filename: str, unsafe: bool
) -> None:
    """``..`` and ``.`` are the ``unsafe=True`` cases (finding 11):
    ``sanitize_blob_filename`` only strips path separators, filesystem-unsafe
    characters, and control characters -- it does not reject a name that
    survives sanitization as a bare dot-only component, so
    ``sanitize_blob_filename("..") == ".."`` verbatim. Landing that as
    ``<prefix>/..`` is exactly the traversal this test otherwise guards
    against; the multi-segment case below is the ORIGINAL regression (it
    sanitizes to a harmless dotted string, not a traversal) and must keep
    succeeding."""
    from pathlib import PurePosixPath

    contract = _contract()
    export_fn = _write_and_export_one(tmp_path, b"payload", suggested_filename)
    _register(_StubExportPlan(export_fn))
    _patch_browser_session(monkeypatch)

    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        if unsafe:
            with pytest.raises(IngestResolutionError, match=_ERR_NO_FILENAME):
                pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]
        else:
            [result] = pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]
            assert result.path.startswith("besrc/2026-08-06/")
            assert ".." not in PurePosixPath(result.path).parts


def test_manifest_json_suggested_filename_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _contract()
    export_fn = _write_and_export_one(tmp_path, b"payload", _MANIFEST_FILENAME)
    _register(_StubExportPlan(export_fn))
    _patch_browser_session(monkeypatch)

    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        with pytest.raises(IngestResolutionError, match=_ERR_RESERVED_FILENAME):
            pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]


def test_manifest_json_payload_filename_template_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _contract(payload_filename_template=_MANIFEST_FILENAME)
    export_fn = _write_and_export_one(tmp_path, b"payload", "report.csv")
    _register(_StubExportPlan(export_fn))
    _patch_browser_session(monkeypatch)

    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        with pytest.raises(IngestResolutionError, match=_ERR_RESERVED_FILENAME):
            pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]


def test_unsanitizable_suggested_filename_raises_with_actionable_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _contract()
    export_fn = _write_and_export_one(tmp_path, b"payload", "///")
    _register(_StubExportPlan(export_fn))
    _patch_browser_session(monkeypatch)

    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        with pytest.raises(IngestResolutionError) as exc_info:
            pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

    message = str(exc_info.value)
    assert _ERR_NO_FILENAME in message
    assert "payload_filename_template" in message


def test_two_files_resolving_to_the_same_name_raise_rather_than_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first_path = tmp_path / "0000.download"
    first_path.write_bytes(b"first")
    second_path = tmp_path / "0001.download"
    second_path.write_bytes(b"second")

    # A genuine collision through sanitize_blob_filename: a run of
    # whitespace of ANY length collapses to a single "_"
    # (filenames.py's `_WHITESPACE_RUN_RE = re.compile(r"\s+")`), so one
    # space and two spaces both resolve to "a_b.json". (Brief note: a
    # tab does NOT collide here as the brief's example suggested -- a tab
    # is an ASCII control character (0x09) and is stripped outright by
    # `_CONTROL_CHARS` before the whitespace-collapse regex ever runs, so
    # "a\tb.json" sanitizes to "ab.json", not "a_b.json"; verified
    # empirically against the shipped `sanitize_blob_filename`.)
    def _export(
        session: Any, partition_key: str, config: dict[str, Any], ctx: IngestContext
    ) -> Iterator[ExportedFile]:
        del session, partition_key, config, ctx
        yield ExportedFile(path=first_path, suggested_filename="a b.json")
        yield ExportedFile(path=second_path, suggested_filename="a  b.json")

    contract = _contract()
    _register(_StubExportPlan(_export, multi_file=True))
    _patch_browser_session(monkeypatch)

    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        with pytest.raises(IngestResolutionError, match=_ERR_DUPLICATE_FILENAME):
            pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

    # Partial-write state is the documented recovery mode: the first file
    # landed before the second one's collision was detected.
    assert "besrc/2026-08-06/a_b.json" in blob.store
    assert blob.store["besrc/2026-08-06/a_b.json"][0] == b"first"


def test_multi_file_plan_lands_distinct_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first_path = tmp_path / "0000.download"
    first_path.write_bytes(b"first")
    second_path = tmp_path / "0001.download"
    second_path.write_bytes(b"second")

    def _export(
        session: Any, partition_key: str, config: dict[str, Any], ctx: IngestContext
    ) -> Iterator[ExportedFile]:
        del session, partition_key, config, ctx
        yield ExportedFile(path=first_path, suggested_filename="alpha.json")
        yield ExportedFile(path=second_path, suggested_filename="beta.json")

    contract = _contract()
    _register(_StubExportPlan(_export, multi_file=True))
    _patch_browser_session(monkeypatch)

    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        results = materialize_with_manifest(pattern, contract, spec, blob, ctx)

    assert len(results) == 2
    assert {r.path for r in results} == {
        "besrc/2026-08-06/alpha.json",
        "besrc/2026-08-06/beta.json",
    }

    from io import BytesIO

    from moncpipelib.ingest.manifest import IngestManifest

    manifest_bytes = blob.store["besrc/2026-08-06/_manifest.json"][0]
    manifest = IngestManifest.read_from(BytesIO(manifest_bytes))
    assert {f.path for f in manifest.files} == {
        "besrc/2026-08-06/alpha.json",
        "besrc/2026-08-06/beta.json",
    }


# ---------------------------------------------------------------------------
# Wiring + containment + config-key errors (#464/#468 findings 2, 12, 13)
# ---------------------------------------------------------------------------


def test_allowed_hosts_from_contract_reach_browser_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Finding 2: the contract's ``allowed_hosts`` must genuinely reach
    ``browser_session(...)`` -- replacing
    ``[str(h) for h in cfg["allowed_hosts"]]`` with ``[]`` at the call site
    left the whole test suite green because every existing test's fake
    discarded the kwargs it was called with (``del kwargs``)."""
    contract = _contract(
        pattern_config=_pattern_config(allowed_hosts=["a.example.gov", "b.example.gov"])
    )
    export_fn = _write_and_export_one(tmp_path, b"payload", "report.csv")
    _register(_StubExportPlan(export_fn))
    captured = _patch_browser_session_capturing(monkeypatch)

    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

    assert captured["allowed_hosts"] == ["a.example.gov", "b.example.gov"]


def test_browser_headless_config_reaches_browser_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Finding 10: ``ingest.browser_export.browser.headless`` must reach
    ``browser_session(headless=...)`` -- hardcoding ``headless=True``
    (or ``False``) at the pattern's call site would leave this contract
    knob a no-op."""
    contract = _contract(pattern_config=_pattern_config(browser={"headless": False}))
    export_fn = _write_and_export_one(tmp_path, b"payload", "report.csv")
    _register(_StubExportPlan(export_fn))
    captured = _patch_browser_session_capturing(monkeypatch)

    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

    assert captured["headless"] is False


def test_fetch_user_agent_config_reaches_browser_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Finding 10: ``ingest.browser_export.fetch.user_agent`` -- a control
    SECURITY.md names explicitly -- must reach
    ``browser_session(user_agent=...)``."""
    contract = _contract(pattern_config=_pattern_config(fetch={"user_agent": "ExampleOrg/1.0"}))
    export_fn = _write_and_export_one(tmp_path, b"payload", "report.csv")
    _register(_StubExportPlan(export_fn))
    captured = _patch_browser_session_capturing(monkeypatch)

    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

    assert captured["user_agent"] == "ExampleOrg/1.0"


def test_browser_download_timeout_config_reaches_browser_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Finding 10: ``ingest.browser_export.browser.download_timeout_s``
    must reach ``browser_session(download_timeout_s=...)`` rather than
    being hardcoded to the default regardless of contract config."""
    contract = _contract(pattern_config=_pattern_config(browser={"download_timeout_s": 12.5}))
    export_fn = _write_and_export_one(tmp_path, b"payload", "report.csv")
    _register(_StubExportPlan(export_fn))
    captured = _patch_browser_session_capturing(monkeypatch)

    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

    assert captured["download_timeout_s"] == 12.5


def test_exported_path_outside_session_download_dir_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Finding 12: a misbehaving or malicious export plan yielding a path
    outside the session's own download directory (e.g. ``/etc/hostname``)
    must not be hashed and uploaded. The real containment logic is pinned
    against the real ``BrowserSession.require_contains`` separately in
    ``tests/test_ingest_browser_session.py``; this test only proves the
    pattern calls it before doing anything else with the path.

    Pre-merge review gate finding 4 strengthened this: asserting only
    ``blob.upload_calls == []`` would still pass even if
    ``require_contains`` ran AFTER ``_hash_file`` -- ``outside_path``
    exists and hashes cleanly, so hashing it first would not itself raise;
    only the (unrelated) upload would still never happen. The
    ``_hash_file`` spy below pins the docstring's actual claim --
    ``require_contains`` runs BEFORE hashing -- by asserting ``_hash_file``
    is never called at all."""
    outside_path = tmp_path / "outside.csv"
    outside_path.write_bytes(b"payload")

    def _export(
        session: Any, partition_key: str, config: dict[str, Any], ctx: IngestContext
    ) -> Iterator[ExportedFile]:
        del session, partition_key, config, ctx
        yield ExportedFile(path=outside_path, suggested_filename="report.csv")

    contract = _contract()
    _register(_StubExportPlan(_export))
    _patch_browser_session(monkeypatch, session=_FakeSession(contains_raises=True))

    hash_calls: list[Path] = []
    real_hash_file = browser_export._hash_file

    def _spy_hash_file(path: Path) -> tuple[str, int]:
        hash_calls.append(path)
        return real_hash_file(path)

    monkeypatch.setattr(browser_export, "_hash_file", _spy_hash_file)

    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        with pytest.raises(IngestResolutionError):
            pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

    assert hash_calls == []
    assert blob.upload_calls == []


def test_missing_export_plan_key_raises_ingest_resolution_error() -> None:
    """Finding 13: a raw ``KeyError`` from ``cfg["export_plan"]`` must not
    escape ``materialize_partition`` -- a hand-built contract (bypassing
    the loader, which requires this key) is the realistic way this dict
    access could ever miss."""
    cfg = _pattern_config()
    del cfg["export_plan"]
    contract = _contract(pattern_config=cfg)
    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        with pytest.raises(IngestResolutionError, match="export_plan") as exc_info:
            pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

    assert not isinstance(exc_info.value, KeyError)


def test_unregistered_export_plan_at_materialize_raises_ingest_resolution_error() -> None:
    """Finding 13: the realistic trigger for this one -- deploy skew where
    the process that materializes has a different export-plan registry
    than the process that validated the contract at load time. Not a
    KeyError substring match: ``get_export_plan`` itself raises
    ``KeyError``, and it must be converted, not merely not-crash by luck."""
    contract = _contract(pattern_config=_pattern_config(export_plan="never_registered_plan"))
    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        with pytest.raises(IngestResolutionError) as exc_info:
            pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

    assert not isinstance(exc_info.value, KeyError)
    message = str(exc_info.value)
    assert "Unknown export plan" in message
    assert "never_registered_plan" in message


def test_missing_allowed_hosts_key_raises_ingest_resolution_error(tmp_path: Path) -> None:
    """Finding 13: a raw ``KeyError`` from ``cfg["allowed_hosts"]`` must not
    escape ``materialize_partition``."""
    cfg = _pattern_config()
    del cfg["allowed_hosts"]
    contract = _contract(pattern_config=cfg)
    export_fn = _write_and_export_one(tmp_path, b"payload", "report.csv")
    _register(_StubExportPlan(export_fn))
    pattern = BrowserExportPattern()
    ctx = _ctx()
    blob = _InMemoryBlob()

    with freeze_time(_FREEZE_DAILY):
        [spec] = pattern.discover_partitions(contract, ctx)
        with pytest.raises(IngestResolutionError, match="allowed_hosts") as exc_info:
            pattern.materialize_partition(contract, spec, blob, ctx)  # type: ignore[arg-type]

    assert not isinstance(exc_info.value, KeyError)


def test_missing_partition_key_raises_ingest_resolution_error() -> None:
    """Finding 13: a raw ``KeyError`` from ``cfg["partition"]`` (inside
    ``_current_boundary_key``, shared by discovery and materialize) must
    not escape either caller."""
    cfg = _pattern_config()
    del cfg["partition"]
    contract = _contract(pattern_config=cfg)
    pattern = BrowserExportPattern()
    ctx = _ctx()

    with pytest.raises(IngestResolutionError, match="partition") as exc_info:
        pattern.discover_partitions(contract, ctx)

    assert not isinstance(exc_info.value, KeyError)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_hash_file_computes_sha_and_size_in_one_pass(tmp_path: Path) -> None:
    data = (b"x" * 5000) + (b"y" * 3000)
    p = tmp_path / "payload.bin"
    p.write_bytes(data)

    sha256, size_bytes = _hash_file(p)

    assert size_bytes == len(data)
    assert sha256 == hashlib.sha256(data).hexdigest()


def test_default_max_first_bytes_check_matches_http_urls() -> None:
    assert browser_export._DEFAULT_MAX_FIRST_BYTES_CHECK == http_urls._DEFAULT_MAX_FIRST_BYTES_CHECK
