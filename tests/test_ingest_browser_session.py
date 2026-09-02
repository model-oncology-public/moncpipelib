"""Tests for the sanctioned playwright surface (#463 Step 2).

Three layers, per the executor brief:

- Layer A: import hygiene, run in subprocesses (hermetic -- a sibling test
  or the ambient CI environment having playwright installed must not taint
  the result).
- Layer B: the chromium probe (:func:`ensure_chromium_available`).
- Layer C: allowlist + lifecycle + download-capture, all against fakes --
  no real browser is available in CI.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

import moncpipelib
from moncpipelib.ingest import _browser
from moncpipelib.ingest._browser import (
    _ERR_CHROMIUM_MISSING,
    _ERR_CONTROL_NOT_FOUND,
    _ERR_PLAYWRIGHT_MISSING,
    _ERR_TIMEOUT,
    BrowserSession,
    _install_host_allowlist,
    browser_session,
    ensure_chromium_available,
)
from moncpipelib.ingest.exceptions import IngestResolutionError
from moncpipelib.ingest.types import IngestContext

# ---------------------------------------------------------------------------
# Shared fakes / helpers
# ---------------------------------------------------------------------------


class _FakeLog:
    def __init__(self) -> None:
        self.info_calls: list[tuple[Any, ...]] = []

    def info(self, *args: Any) -> None:
        self.info_calls.append(args)


def _fake_ctx() -> IngestContext:
    return IngestContext(log=_FakeLog())


# ---------------------------------------------------------------------------
# Layer A -- import hygiene (subprocess, hermetic)
# ---------------------------------------------------------------------------

_MONCPIPELIB_SRC_ROOT = str(Path(moncpipelib.__file__).parent.parent)


def _assert_import_does_not_pull_in_playwright(module_name: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; import {module_name}; sys.exit(1 if 'playwright' in sys.modules else 0)",
        ],
        env={"PYTHONPATH": _MONCPIPELIB_SRC_ROOT},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"importing {module_name!r} pulled in playwright -- "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_importing_ingest_does_not_import_playwright() -> None:
    _assert_import_does_not_pull_in_playwright("moncpipelib.ingest")


def test_importing_browser_module_does_not_import_playwright() -> None:
    _assert_import_does_not_pull_in_playwright("moncpipelib.ingest._browser")


def test_importing_export_plans_does_not_import_playwright() -> None:
    _assert_import_does_not_pull_in_playwright("moncpipelib.ingest.export_plans")


# ---------------------------------------------------------------------------
# Layer B -- the chromium probe
# ---------------------------------------------------------------------------


def test_ensure_chromium_available_raises_when_playwright_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    with pytest.raises(IngestResolutionError, match=re.escape(_ERR_PLAYWRIGHT_MISSING)):
        ensure_chromium_available()


def test_ensure_chromium_available_raises_when_chromium_binary_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The naturally-reachable CI case (package installed via `uv sync
    --all-extras`, no `playwright install chromium` run) -- made
    deterministic in every environment, chromium binary present or not,
    by monkeypatching the resolved executable path to one that does not
    exist rather than relying on the ambient environment lacking a
    chromium binary.  A ``pytest.skip`` here would leave D2's probe --
    the load-bearing "fail early, never mid-run" control -- silently
    untested the moment anyone runs ``playwright install chromium`` in
    CI.
    """
    missing_executable = tmp_path / "no-such-chromium-binary"

    class _FakeChromiumHandle:
        executable_path = str(missing_executable)

    class _FakePwHandle:
        chromium = _FakeChromiumHandle()

    class _FakeSyncPlaywrightCM:
        def __enter__(self) -> _FakePwHandle:
            return _FakePwHandle()

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
            return False

    fake_module = types.ModuleType("playwright.sync_api")
    fake_module.sync_playwright = lambda: _FakeSyncPlaywrightCM()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_module)

    assert not missing_executable.exists()

    with pytest.raises(IngestResolutionError, match=re.escape(_ERR_CHROMIUM_MISSING)):
        ensure_chromium_available()
    with pytest.raises(IngestResolutionError, match="playwright install chromium"):
        ensure_chromium_available()


def test_probe_error_messages_are_distinguishable() -> None:
    assert _ERR_PLAYWRIGHT_MISSING != _ERR_CHROMIUM_MISSING
    assert not _ERR_PLAYWRIGHT_MISSING.startswith(_ERR_CHROMIUM_MISSING)
    assert not _ERR_CHROMIUM_MISSING.startswith(_ERR_PLAYWRIGHT_MISSING)


# ---------------------------------------------------------------------------
# Layer C -- allowlist + lifecycle + download capture, against fakes
# ---------------------------------------------------------------------------


class _FakeRequest:
    def __init__(self, url: str) -> None:
        self.url = url


class _FakeRoute:
    def __init__(self, url: str) -> None:
        self.request = _FakeRequest(url)
        self.continued = False
        self.aborted = False

    def continue_(self) -> None:
        self.continued = True

    def abort(self) -> None:
        self.aborted = True


class _FakeWsImplObj:
    def __init__(self) -> None:
        self.close_called = False

    async def close(self) -> None:
        self.close_called = True


class _FakeWebSocketRoute:
    def __init__(self, url: str) -> None:
        self.url = url
        self.connected = False
        self._impl_obj = _FakeWsImplObj()

    def connect_to_server(self) -> None:
        self.connected = True


class _FakeAllowlistContext:
    def __init__(self) -> None:
        self.routes: list[tuple[str, Any]] = []
        self.ws_routes: list[tuple[str, Any]] = []

    def route(self, pattern: str, handler: Any) -> None:
        self.routes.append((pattern, handler))

    def route_web_socket(self, pattern: str, handler: Any) -> None:
        self.ws_routes.append((pattern, handler))


def _install_and_get_handlers(allowed: frozenset[str]) -> tuple[Any, Any]:
    context = _FakeAllowlistContext()
    _install_host_allowlist(context, allowed, _fake_ctx())
    assert len(context.routes) == 1
    assert len(context.ws_routes) == 1
    return context.routes[0][1], context.ws_routes[0][1]


def test_allowlist_continues_request_to_allowed_host() -> None:
    route_handler, _ = _install_and_get_handlers(frozenset({"340bopais.hrsa.gov"}))
    route = _FakeRoute("https://340bopais.hrsa.gov/reports")

    route_handler(route)

    assert route.continued is True
    assert route.aborted is False


def test_allowlist_aborts_request_to_disallowed_host() -> None:
    route_handler, _ = _install_and_get_handlers(frozenset({"340bopais.hrsa.gov"}))
    route = _FakeRoute("https://evil.example/x.js")

    route_handler(route)

    assert route.aborted is True
    assert route.continued is False


def test_allowlist_aborts_disallowed_subresource_and_redirect_target() -> None:
    route_handler, _ = _install_and_get_handlers(frozenset({"340bopais.hrsa.gov"}))

    subresource = _FakeRoute("https://cdn.example/app.js")
    redirect_target = _FakeRoute("https://tracker.example/r?u=x")

    route_handler(subresource)
    route_handler(redirect_target)

    assert subresource.aborted is True
    assert redirect_target.aborted is True


def test_allowlist_matching_is_case_insensitive_and_port_independent() -> None:
    """Pins port-independence genuinely; the case-insensitivity half of
    this test's name is actually pinning ``urlsplit(...).hostname``
    lowercasing the incoming URL, not any code in this library (finding
    8) -- see ``test_browser_session_lowercases_configured_allowed_hosts``
    for the test that pins THIS library's own lowering of the
    *configured* allowlist, which is the half that matters."""
    route_handler, _ = _install_and_get_handlers(frozenset({"340bopais.hrsa.gov"}))
    route = _FakeRoute("https://340BOPAIS.HRSA.GOV:443/reports")

    route_handler(route)

    assert route.continued is True
    assert route.aborted is False


def test_allowlist_does_not_suffix_match() -> None:
    route_handler, _ = _install_and_get_handlers(frozenset({"340bopais.hrsa.gov"}))
    route = _FakeRoute("https://evil-340bopais.hrsa.gov.attacker.example/")

    route_handler(route)

    assert route.aborted is True
    assert route.continued is False


def test_allowlist_request_with_trailing_dot_matches_configured_host_without_it() -> None:
    """Pre-merge review gate finding 1: a request naming ``hrsa.gov.`` --
    the valid DNS root label, preserved verbatim by both ``urlsplit`` and
    a real Chromium -- must still be recognised as the SAME host as a
    configured ``hrsa.gov`` allowlist entry. Without this, the trailing
    dot form of ANY reserved hostname (e.g. ``localhost.``, rejected at
    contract load per ``contracts/loader.py``) would sail straight past
    this comparison at runtime even though the loader's rejection exists
    to prevent exactly that host from ever reaching this code."""
    route_handler, _ = _install_and_get_handlers(frozenset({"hrsa.gov"}))
    route = _FakeRoute("https://hrsa.gov./reports")

    route_handler(route)

    assert route.continued is True
    assert route.aborted is False


def test_allowlist_continues_non_http_schemes() -> None:
    route_handler, _ = _install_and_get_handlers(frozenset())

    for url in (
        "blob:https://340bopais.hrsa.gov/uuid",
        "data:text/plain,x",
        "about:blank",
    ):
        route = _FakeRoute(url)
        route_handler(route)
        assert route.continued is True, url
        assert route.aborted is False, url


def test_websocket_route_connects_to_server_for_allowed_host() -> None:
    _, ws_handler = _install_and_get_handlers(frozenset({"340bopais.hrsa.gov"}))
    ws = _FakeWebSocketRoute("wss://340bopais.hrsa.gov/_blazor")

    asyncio.run(ws_handler(ws))

    assert ws.connected is True
    assert ws._impl_obj.close_called is False


def test_websocket_route_closes_for_disallowed_host() -> None:
    _, ws_handler = _install_and_get_handlers(frozenset({"340bopais.hrsa.gov"}))
    ws = _FakeWebSocketRoute("wss://evil.example/socket")

    asyncio.run(ws_handler(ws))

    assert ws._impl_obj.close_called is True
    assert ws.connected is False


def test_websocket_route_with_trailing_dot_matches_configured_host_without_it() -> None:
    """Pre-merge review gate finding 1, WebSocket half: same normalization
    as ``test_allowlist_request_with_trailing_dot_matches_configured_host_without_it``,
    for the ``route_web_socket`` handler."""
    _, ws_handler = _install_and_get_handlers(frozenset({"340bopais.hrsa.gov"}))
    ws = _FakeWebSocketRoute("wss://340bopais.hrsa.gov./_blazor")

    asyncio.run(ws_handler(ws))

    assert ws.connected is True
    assert ws._impl_obj.close_called is False


def test_registered_route_pattern_matches_ws_and_http_via_playwrights_own_matcher() -> None:
    """Pre-merge review gate finding 2: playwright's own
    ``BrowserContext._on_web_socket_route`` calls ``connect_to_server()``
    (GRANTS) when no handler matches -- fail-OPEN by default. The glob
    this module registers (``"**/*"``, read here off the real fake
    ``_FakeAllowlistContext`` rather than hand-copied, so narrowing it in
    ``_browser.py`` is picked up automatically) is the only thing
    preventing that default from applying to every WebSocket upgrade. Every
    OTHER allowlist test in this module calls the registered handler
    directly and would stay green even if the glob were narrowed to
    something that silently stops matching WebSocket URLs (e.g.
    ``"https://**"``, which reads like a plausible "tighten this" edit) --
    this test instead drives playwright's OWN glob-matching engine against
    the literal pattern ``_install_host_allowlist`` hands to
    ``context.route`` / ``context.route_web_socket``, so a narrowed glob
    fails HERE even though every fake-based test above still passes.

    ``url_matches`` is not part of playwright's public ``sync_api`` /
    ``async_api`` surface -- it lives in the private ``_impl._helper``
    module. Importing it couples this test to playwright's internal
    module layout across version bumps; if a future release moves or
    renames it, this TEST breaks (production code does not depend on it),
    and the fix is to find its new home, not to delete the assertion.
    """
    from playwright._impl._helper import url_matches

    context = _FakeAllowlistContext()
    _install_host_allowlist(context, frozenset(), _fake_ctx())
    (route_pattern, _) = context.routes[0]
    (ws_pattern, _) = context.ws_routes[0]

    for pattern, url in (
        (route_pattern, "https://340bopais.hrsa.gov/reports"),
        (ws_pattern, "wss://340bopais.hrsa.gov/_blazor"),
        (ws_pattern, "ws://340bopais.hrsa.gov/_blazor"),
    ):
        assert url_matches(None, url, pattern) is True, (pattern, url)


def test_session_exposes_no_raw_browser_handle() -> None:
    # `require_contains` (finding 12) is a deliberate sixth addition: it
    # raises rather than returning a boolean the caller could forget to
    # check (consistent with every other method's failure-signalling
    # contract), and it is not an accessor for the underlying
    # page/context/browser/playwright handle the module docstring forbids
    # exposing -- it is how the `browser_export` pattern verifies an
    # `ExportedFile.path` a plan yields actually lives inside this
    # session's own download directory before hashing/uploading it.
    public = {n for n in dir(BrowserSession) if not n.startswith("_")}
    assert public == {
        "navigate",
        "expect_control",
        "click",
        "select_option",
        "click_and_await_download",
        "require_contains",
    }


# --- fakes for the browser_session() lifecycle ---


class _FakePage:
    def __init__(self) -> None:
        self.default_timeout_ms: float | None = None
        self.default_navigation_timeout_ms: float | None = None

    def set_default_timeout(self, timeout: float) -> None:
        self.default_timeout_ms = timeout

    def set_default_navigation_timeout(self, timeout: float) -> None:
        self.default_navigation_timeout_ms = timeout


class _FakeLifecycleContext:
    def __init__(self) -> None:
        self.closed = False
        self.routes: list[tuple[str, Any]] = []
        self.ws_routes: list[tuple[str, Any]] = []
        self.new_context_kwargs: dict[str, Any] = {}
        self.page: _FakePage | None = None

    def route(self, pattern: str, handler: Any) -> None:
        self.routes.append((pattern, handler))

    def route_web_socket(self, pattern: str, handler: Any) -> None:
        self.ws_routes.append((pattern, handler))

    def new_page(self) -> _FakePage:
        self.page = _FakePage()
        return self.page

    def close(self) -> None:
        self.closed = True


class _FakeLifecycleBrowser:
    def __init__(self) -> None:
        self.closed = False
        self.contexts: list[_FakeLifecycleContext] = []

    def new_context(self, **kwargs: Any) -> _FakeLifecycleContext:
        ctx = _FakeLifecycleContext()
        ctx.new_context_kwargs = dict(kwargs)
        self.contexts.append(ctx)
        return ctx

    def close(self) -> None:
        self.closed = True


class _FakeChromium:
    def __init__(
        self,
        launch_calls: list[dict[str, Any]],
        browser: _FakeLifecycleBrowser | None,
        raise_on_launch: Exception | None = None,
    ) -> None:
        self._launch_calls = launch_calls
        self._browser = browser
        self._raise_on_launch = raise_on_launch

    def launch(self, **kwargs: Any) -> _FakeLifecycleBrowser:
        self._launch_calls.append(kwargs)
        if self._raise_on_launch is not None:
            raise self._raise_on_launch
        assert self._browser is not None
        return self._browser


class _FakePlaywrightHandle:
    def __init__(self, chromium: _FakeChromium) -> None:
        self.chromium = chromium


class _FakeSyncPlaywrightCM:
    def __init__(self, chromium: _FakeChromium) -> None:
        self._chromium = chromium

    def __enter__(self) -> _FakePlaywrightHandle:
        return _FakePlaywrightHandle(self._chromium)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return False


def _install_fake_playwright(monkeypatch: pytest.MonkeyPatch, chromium: _FakeChromium) -> None:
    monkeypatch.setattr(_browser, "ensure_chromium_available", lambda: None)
    fake_module = types.ModuleType("playwright.sync_api")
    fake_module.sync_playwright = lambda: _FakeSyncPlaywrightCM(chromium)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_module)


def test_session_tempdir_removed_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    launch_calls: list[dict[str, Any]] = []
    browser = _FakeLifecycleBrowser()
    chromium = _FakeChromium(launch_calls, browser)
    _install_fake_playwright(monkeypatch, chromium)

    captured_tempdir: Path | None = None
    with browser_session(allowed_hosts=["example.com"], ctx=_fake_ctx()) as session:
        captured_tempdir = session._download_dir.parent
        assert captured_tempdir.exists()

    assert captured_tempdir is not None
    assert not captured_tempdir.exists()
    assert browser.closed is True
    assert browser.contexts[0].closed is True


def test_session_tempdir_removed_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    launch_calls: list[dict[str, Any]] = []
    browser = _FakeLifecycleBrowser()
    chromium = _FakeChromium(launch_calls, browser)
    _install_fake_playwright(monkeypatch, chromium)

    captured_tempdir: Path | None = None

    class _Boom(Exception):
        pass

    with (
        pytest.raises(_Boom),
        browser_session(allowed_hosts=["example.com"], ctx=_fake_ctx()) as session,
    ):
        captured_tempdir = session._download_dir.parent
        raise _Boom("boom")

    assert captured_tempdir is not None
    assert not captured_tempdir.exists()
    assert browser.closed is True
    assert browser.contexts[0].closed is True


def test_session_tempdir_removed_when_launch_fails_midway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch_calls: list[dict[str, Any]] = []
    chromium = _FakeChromium(
        launch_calls, browser=None, raise_on_launch=RuntimeError("no chromium")
    )
    _install_fake_playwright(monkeypatch, chromium)

    with (
        pytest.raises(RuntimeError, match="no chromium"),
        browser_session(allowed_hosts=["example.com"], ctx=_fake_ctx()),
    ):
        pytest.fail("should not reach the with-body: launch() raised first")

    assert len(launch_calls) == 1
    # No Browser object was ever constructed (launch raised before returning
    # one), so there is nothing whose .close() could have been called --
    # the ExitStack only unwinds what it actually registered.
    tempdir = Path(launch_calls[0]["downloads_path"]).parent
    assert not tempdir.exists()


# ---------------------------------------------------------------------------
# Wiring + config plumbing (#464/#468 findings 1, 2, 8, 10)
# ---------------------------------------------------------------------------


def test_browser_session_installs_allowlist_handlers_on_the_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins the WIRING, not just the handler logic (finding 2): mutating
    ``_install_host_allowlist`` to a no-op ``pass`` must fail a test.  Every
    other allowlist test drives ``_install_host_allowlist`` directly; this
    is the one that proves ``browser_session()`` actually calls it on the
    real context."""
    launch_calls: list[dict[str, Any]] = []
    browser = _FakeLifecycleBrowser()
    chromium = _FakeChromium(launch_calls, browser)
    _install_fake_playwright(monkeypatch, chromium)

    with browser_session(allowed_hosts=["example.com"], ctx=_fake_ctx()):
        pass

    [context] = browser.contexts
    assert len(context.routes) == 1
    assert len(context.ws_routes) == 1


def test_browser_session_lowercases_configured_allowed_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins ``_browser.py``'s own lowering of the *configured* allowlist
    (finding 8) as distinct from ``urlsplit(...).hostname`` already
    lowercasing the *incoming* URL's host (which
    ``test_allowlist_matching_is_case_insensitive_and_port_independent``
    actually pins -- that test passes even with no lowering in this
    library's own code, because ``urllib`` already did the work).  The
    loader does not require lowercase ``allowed_hosts`` entries, so a
    contract writing ``allowed_hosts: ["340BOPAIS.HRSA.GOV"]`` depends
    entirely on this line."""
    launch_calls: list[dict[str, Any]] = []
    browser = _FakeLifecycleBrowser()
    chromium = _FakeChromium(launch_calls, browser)
    _install_fake_playwright(monkeypatch, chromium)

    with browser_session(allowed_hosts=["340BOPAIS.HRSA.GOV"], ctx=_fake_ctx()):
        pass

    [context] = browser.contexts
    [(_, route_handler)] = context.routes
    route = _FakeRoute("https://340bopais.hrsa.gov/reports")

    route_handler(route)

    assert route.continued is True
    assert route.aborted is False


def test_browser_session_strips_trailing_dot_from_configured_allowed_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-merge review gate finding 1: ``browser_session``'s own
    normalization of the *configured* allowlist must strip a trailing
    ``.`` -- the valid DNS root label -- the same way it already
    lowercases (see ``test_browser_session_lowercases_configured_allowed_hosts``).
    Without this, an allowlist entry authored as ``340bopais.hrsa.gov.``
    could never match a request to ``340bopais.hrsa.gov`` -- a latent
    fail-CLOSED bug in the safe direction, but a silent one: nothing would
    tell an operator why every request was being blocked."""
    launch_calls: list[dict[str, Any]] = []
    browser = _FakeLifecycleBrowser()
    chromium = _FakeChromium(launch_calls, browser)
    _install_fake_playwright(monkeypatch, chromium)

    with browser_session(allowed_hosts=["340bopais.hrsa.gov."], ctx=_fake_ctx()):
        pass

    [context] = browser.contexts
    [(_, route_handler)] = context.routes
    route = _FakeRoute("https://340bopais.hrsa.gov/reports")

    route_handler(route)

    assert route.continued is True
    assert route.aborted is False


def test_browser_session_blocks_service_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding 1: Service Workers bypass ``BrowserContext.route()`` entirely
    (playwright's own ``route()`` docstring recommends ``service_workers=
    "block"`` for exactly this reason) -- a page registering one would
    defeat the one network-interception control this pattern's design
    rests on."""
    launch_calls: list[dict[str, Any]] = []
    browser = _FakeLifecycleBrowser()
    chromium = _FakeChromium(launch_calls, browser)
    _install_fake_playwright(monkeypatch, chromium)

    with browser_session(allowed_hosts=["example.com"], ctx=_fake_ctx()):
        pass

    assert browser.contexts[0].new_context_kwargs["service_workers"] == "block"


def test_browser_session_passes_accept_downloads_true(monkeypatch: pytest.MonkeyPatch) -> None:
    launch_calls: list[dict[str, Any]] = []
    browser = _FakeLifecycleBrowser()
    chromium = _FakeChromium(launch_calls, browser)
    _install_fake_playwright(monkeypatch, chromium)

    with browser_session(allowed_hosts=["example.com"], ctx=_fake_ctx()):
        pass

    assert browser.contexts[0].new_context_kwargs["accept_downloads"] is True


def test_browser_session_passes_configured_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    launch_calls: list[dict[str, Any]] = []
    browser = _FakeLifecycleBrowser()
    chromium = _FakeChromium(launch_calls, browser)
    _install_fake_playwright(monkeypatch, chromium)

    with browser_session(
        allowed_hosts=["example.com"], ctx=_fake_ctx(), user_agent="ExampleOrg/1.0"
    ):
        pass

    assert browser.contexts[0].new_context_kwargs["user_agent"] == "ExampleOrg/1.0"


def test_browser_session_omits_user_agent_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch_calls: list[dict[str, Any]] = []
    browser = _FakeLifecycleBrowser()
    chromium = _FakeChromium(launch_calls, browser)
    _install_fake_playwright(monkeypatch, chromium)

    with browser_session(allowed_hosts=["example.com"], ctx=_fake_ctx()):
        pass

    assert "user_agent" not in browser.contexts[0].new_context_kwargs


def test_browser_session_passes_headless_flag_through_to_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch_calls: list[dict[str, Any]] = []
    browser = _FakeLifecycleBrowser()
    chromium = _FakeChromium(launch_calls, browser)
    _install_fake_playwright(monkeypatch, chromium)

    with browser_session(allowed_hosts=["example.com"], ctx=_fake_ctx(), headless=False):
        pass

    assert launch_calls[0]["headless"] is False


def test_browser_session_navigation_timeout_reaches_the_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch_calls: list[dict[str, Any]] = []
    browser = _FakeLifecycleBrowser()
    chromium = _FakeChromium(launch_calls, browser)
    _install_fake_playwright(monkeypatch, chromium)

    with browser_session(allowed_hosts=["example.com"], ctx=_fake_ctx(), navigation_timeout_s=42.0):
        pass

    page = browser.contexts[0].page
    assert page is not None
    assert page.default_timeout_ms == 42_000.0
    assert page.default_navigation_timeout_ms == 42_000.0


def test_browser_session_download_timeout_reaches_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch_calls: list[dict[str, Any]] = []
    browser = _FakeLifecycleBrowser()
    chromium = _FakeChromium(launch_calls, browser)
    _install_fake_playwright(monkeypatch, chromium)

    with browser_session(
        allowed_hosts=["example.com"], ctx=_fake_ctx(), download_timeout_s=123.0
    ) as session:
        assert session._download_timeout_s == 123.0


# --- fakes for click_and_await_download() ---


class _FakePlaywrightDownloadError(Exception):
    """Stands in for ``playwright.sync_api.Error``: the type nothing in
    this library recognises, which must never escape
    ``click_and_await_download`` uncaught (#464/#468 findings 5+6)."""


class _FakeDownload:
    def __init__(self, suggested_filename: str, failure_msg: str | None = None) -> None:
        self.suggested_filename = suggested_filename
        self._failure_msg = failure_msg
        self.saved_to: Path | None = None

    def save_as(self, dest: Path) -> None:
        """Mirrors real playwright: ``save_as`` resolves the download
        artifact server-side and raises FIRST when the download failed --
        it does not return cleanly and leave ``failure()`` to report the
        problem afterwards.  The previous version of this fake returned
        cleanly here even with ``failure_msg`` set, certifying a code path
        real playwright cannot reach."""
        if self._failure_msg is not None:
            raise _FakePlaywrightDownloadError(self._failure_msg)
        self.saved_to = dest
        dest.write_bytes(b"data")

    def failure(self) -> str | None:
        return self._failure_msg


class _FakeDownloadLocator:
    def __init__(self, click_error: Exception | None = None) -> None:
        self._click_error = click_error
        self.clicked = False

    def click(self, *, timeout: float) -> None:
        del timeout
        if self._click_error is not None:
            raise self._click_error
        self.clicked = True


class _FakeExpectDownloadCM:
    """Mirrors playwright's ``EventContextManager`` timing (verified against
    a real chromium 1.62.0): ``.value`` resolves -- or the download-wait
    timeout raises -- on ``__exit__``, not when ``.value`` is later read;
    an exception already propagating from inside the ``with`` body is left
    alone (no re-check of the pending event)."""

    def __init__(self, download: _FakeDownload | None, raise_on_exit: Exception | None) -> None:
        self._download = download
        self._raise_on_exit = raise_on_exit
        self.value: Any = None

    def __enter__(self) -> _FakeExpectDownloadCM:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_val is not None:
            return False
        if self._raise_on_exit is not None:
            raise self._raise_on_exit
        self.value = self._download
        return False


class _FakeDownloadPage:
    def __init__(self, locator: _FakeDownloadLocator, expect_cm: _FakeExpectDownloadCM) -> None:
        self._locator = locator
        self._expect_cm = expect_cm

    def get_by_role(self, role: str, *, name: str, exact: bool) -> _FakeDownloadLocator:
        del role, name, exact
        return self._locator

    def expect_download(self, *, timeout: float) -> _FakeExpectDownloadCM:
        del timeout
        return self._expect_cm


def _make_session(page: Any, download_dir: Path) -> BrowserSession:
    return BrowserSession(
        page=page,
        download_dir=download_dir,
        navigation_timeout_s=5.0,
        download_timeout_s=5.0,
    )


# ---------------------------------------------------------------------------
# require_contains() -- the containment boundary check (#464/#468 finding 12)
# ---------------------------------------------------------------------------


def test_require_contains_does_not_raise_for_a_path_inside_the_download_dir(
    tmp_path: Path,
) -> None:
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    session = _make_session(page=object(), download_dir=download_dir)

    inside = download_dir / "0000.download"
    inside.write_bytes(b"payload")

    session.require_contains(inside)  # must not raise


def test_require_contains_raises_for_a_path_outside_the_download_dir(tmp_path: Path) -> None:
    """A misbehaving or malicious export plan yielding an arbitrary
    pod-local path (e.g. ``/etc/hostname``) must not be treated as
    contained -- ``BrowserSession`` is framed as containment (D5), so
    this must be verifiable rather than merely conventional."""
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    session = _make_session(page=object(), download_dir=download_dir)

    outside = tmp_path / "not-a-download.csv"
    outside.write_bytes(b"payload")

    with pytest.raises(IngestResolutionError):
        session.require_contains(outside)


def test_require_contains_raises_for_a_traversal_path_resolving_outside(
    tmp_path: Path,
) -> None:
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    session = _make_session(page=object(), download_dir=download_dir)

    (tmp_path / "escaped.csv").write_bytes(b"payload")
    traversal = download_dir / ".." / "escaped.csv"

    with pytest.raises(IngestResolutionError):
        session.require_contains(traversal)


def test_require_contains_raises_for_the_download_dir_itself(tmp_path: Path) -> None:
    """Pre-merge review gate finding 4: the download directory itself is
    not a file. ``resolved.relative_to(download_dir)`` returns
    ``Path(".")`` for it rather than raising, so the containment check
    alone let it through -- this pinned the PERMISSIVE behavior before
    the fix (see git history); it now pins the corrected contract:
    ``require_contains`` must reject anything that is not an existing
    regular file inside the download dir, the directory itself included.
    Reachable because ``ExportedFile`` is in ``moncpipelib.ingest.__all__``,
    so a plan (malicious or merely buggy) can construct one with any
    path, including the download dir itself -- which would otherwise
    reach ``browser_export.py``'s ``_hash_file`` and raise a raw
    ``IsADirectoryError``."""
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    session = _make_session(page=object(), download_dir=download_dir)

    with pytest.raises(IngestResolutionError):
        session.require_contains(download_dir)


def test_require_contains_raises_for_a_nonexistent_path_inside_the_download_dir(
    tmp_path: Path,
) -> None:
    """Finding 4: containment alone is purely lexical about existence -- a
    path that resolves inside the download dir but was never actually
    written there must still be rejected, rather than reaching
    ``_hash_file`` and raising a raw ``FileNotFoundError``."""
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    session = _make_session(page=object(), download_dir=download_dir)

    missing = download_dir / "0000.download"
    assert not missing.exists()

    with pytest.raises(IngestResolutionError):
        session.require_contains(missing)


def test_require_contains_raises_for_a_subdirectory_inside_the_download_dir(
    tmp_path: Path,
) -> None:
    """Finding 4: a subdirectory resolves inside the download dir (passes
    containment) but is not a file either."""
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    nested = download_dir / "nested"
    nested.mkdir()
    session = _make_session(page=object(), download_dir=download_dir)

    with pytest.raises(IngestResolutionError):
        session.require_contains(nested)


def test_click_and_await_download_saves_to_session_controlled_filename(tmp_path: Path) -> None:
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    download = _FakeDownload(suggested_filename="../../escape.json")
    locator = _FakeDownloadLocator()
    page = _FakeDownloadPage(locator, _FakeExpectDownloadCM(download, None))
    session = _make_session(page, download_dir)

    result = session.click_and_await_download(role="button", name="Export")

    assert result.path.parent == download_dir
    assert re.fullmatch(r"\d{4}\.download", result.path.name)
    assert result.suggested_filename == "../../escape.json"


def test_click_and_await_download_maps_click_failure_to_control_not_found(
    tmp_path: Path,
) -> None:
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    locator = _FakeDownloadLocator(click_error=RuntimeError("no such element"))
    page = _FakeDownloadPage(locator, _FakeExpectDownloadCM(None, None))
    session = _make_session(page, download_dir)

    with pytest.raises(IngestResolutionError, match=re.escape(_ERR_CONTROL_NOT_FOUND)) as exc_info:
        session.click_and_await_download(role="button", name="Export")

    message = str(exc_info.value)
    assert "role='button'" in message
    assert "name='Export'" in message


def test_click_and_await_download_maps_download_wait_timeout_to_timeout_error(
    tmp_path: Path,
) -> None:
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    locator = _FakeDownloadLocator()
    page = _FakeDownloadPage(
        locator, _FakeExpectDownloadCM(None, TimeoutError("no download event"))
    )
    session = _make_session(page, download_dir)

    with pytest.raises(IngestResolutionError, match=re.escape(_ERR_TIMEOUT)) as exc_info:
        session.click_and_await_download(role="button", name="Export")

    assert _ERR_CONTROL_NOT_FOUND not in str(exc_info.value)


def test_click_and_await_download_raises_when_download_reports_failure(tmp_path: Path) -> None:
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    download = _FakeDownload(suggested_filename="report.csv", failure_msg="net::ERR_ABORTED")
    locator = _FakeDownloadLocator()
    page = _FakeDownloadPage(locator, _FakeExpectDownloadCM(download, None))
    session = _make_session(page, download_dir)

    with pytest.raises(IngestResolutionError, match=re.escape(_ERR_TIMEOUT)) as exc_info:
        session.click_and_await_download(role="button", name="Export")

    assert "net::ERR_ABORTED" in str(exc_info.value)
