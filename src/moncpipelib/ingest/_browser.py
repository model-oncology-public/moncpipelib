"""Sanctioned playwright surface for the ``browser_export`` ingest pattern (#463).

This module is the ONLY place in the library allowed to touch playwright.
Export plans (:mod:`moncpipelib.ingest.export_plans`) never construct a
browser, a playwright context, or an ``httpx.Client`` of their own -- every
interaction with the driving source goes through the
:class:`BrowserSession` yielded by :func:`browser_session`.

**playwright is imported LAZILY, inside function bodies only** -- never at
module import, never at discovery.  This is the load-bearing constraint of
#463 D1: ``patterns/__init__.py::_register_builtin_patterns()`` imports
``browser_export`` at ``import moncpipelib.ingest`` time (Step 3), so any
module-level playwright import in that chain would break
``Definitions(...)`` construction for every consumer -- including ones that
only write to Postgres and never touch a browser.  See
``tests/test_ingest_browser_session.py``'s subprocess-based import-hygiene
tests for the enforced guard.

**Typing.**  Every playwright object (the ``sync_playwright()`` handle, the
``Browser``, ``BrowserContext``, ``Page``, ``Route``, ``WebSocketRoute``, and
``Download``) is annotated ``Any`` -- deliberately.  playwright ships no
``py.typed`` marker for a codebase that imports it lazily and conditionally
the way this one does, and ``mypy src`` must behave the same whether or not
the optional ``browser`` extra happens to be installed in the environment it
runs in; a ``TYPE_CHECKING``-only playwright import is exactly the kind of
change a future editor could make that quietly reintroduces that coupling.
Every method here that returns something derived from a playwright value
coerces it explicitly at the boundary (e.g.
``str(download.suggested_filename or "")``) so ``warn_return_any`` has
nothing to flag.

**Containment, not policy (D5/D11).**  :class:`BrowserSession` exposes
exactly six public methods and nothing else: no accessor for the
underlying page, context, browser, or playwright handle, and no
``screenshot`` method.  ``require_contains`` is the one method that is a
boundary *check* rather than a browsing action -- it raises unless a path
resolves to inside the session's own download directory, so the pattern
can verify (not just trust) that an ``ExportedFile.path`` a plan yields
actually lives where the session put it, before hashing or uploading it.
Locating controls is by ARIA role + accessible name only -- there is no
``css=`` escape hatch (D6), so a DevExpress-generated CSS class can never
be used to reach a control.  This is enforced by a structural test (the
allowed *set* of public members), not a hand-listed deny-list.

**Failure signalling.**  Every method returns ``None`` or
:class:`~moncpipelib.ingest.export_plans.ExportedFile` and signals failure
by raising :class:`~moncpipelib.ingest.exceptions.IngestResolutionError`.
No method returns a status code, a boolean, or a sentinel: a boolean
``False`` from ``click()`` would be indistinguishable at a call site from a
call nobody checked, and a plan that ignores it would proceed to
``click_and_await_download`` and produce a *timeout* error naming the wrong
control.  Raising is the only failure value that cannot read as a valid
state.

**Host allowlist scope (D5).**  :func:`_install_host_allowlist` governs
HTTP(S) requests and WebSocket connections only.  Any other URL scheme
(``blob:``, ``data:``, ``about:``, ``chrome-error:``, ...) is continued
unconditionally: these never reach the browser's network stack (confirmed
empirically against a real chromium 1.62.0 -- a route registered with the
broadest possible catch-all glob is never even invoked for a ``data:`` or
``about:blank`` navigation), and a ``blob:`` URL is the exact mechanism the
driving source is expected to use for its own download, per the design
doc's D5.  This scope limit is intentional and is documented again in
``SECURITY.md`` (Step 4) -- it is not an oversight.  This allowlist itself
is reclassified, per the #463 egress-allowlist reframe
(``docs/migrations/20260807_463-egress-allowlist-reframe.md`` and
``SECURITY.md``'s ``allowed_hosts`` section), as a contract-authoring
mistake-catcher rather than a security boundary: it cannot decide where a
name resolves, and the network egress boundary, where one is deployed, is
deployment-provided, not this check.

**Service Workers are disabled outright, not merely uncovered.**  Unlike
the scheme carve-outs above, a page-registered Service Worker is not a
documented scope limit of this allowlist -- it is a way for a page's own
requests to bypass this navigation/subresource check entirely, rather
than merely fall outside its documented scope: playwright's own
``browser_context.route()`` docstring states verbatim that
Service-Worker-intercepted requests are never seen by ``route()``
handlers, and recommends ``service_workers="block"`` when using request
interception (as this pattern does).  :func:`browser_session` therefore
passes ``service_workers="block"`` to ``browser.new_context(...)`` rather
than leaving it to the default, so a page cannot register one to route
around this check.

**A confirmed playwright-python sync-API limitation (WebSocket deny path).**
The design calls for the WebSocket route handler to call ``ws.close()`` on a
disallowed host, so the page sees an explicit close rather than a silently
mocked-open connection that would hang the driving source's realtime
circuit forever.  Empirically (playwright-python 1.62.0), calling the
public, sync-wrapped ``WebSocketRoute.close()`` **from inside a
``route_web_socket`` handler deadlocks the whole browser session** while a
``navigate()``/``click()`` call is in flight: unlike HTTP ``route()``
handlers (which playwright-python runs on a dedicated greenlet precisely so
they can make further blocking sync calls), a ``route_web_socket`` handler
is invoked directly on the dispatcher fiber, so ``close()``'s blocking
wait-for-acknowledgement tries to switch the dispatcher fiber to itself and
never completes.  ``WebSocketRoute.connect_to_server()`` does not have this
problem (it is fire-and-forget and never waits for a response), which is
why the allow branch is safe as a plain synchronous call.  The workaround
below registers an ``async def`` handler -- ``WebSocketRouteHandler.handle``
explicitly supports awaiting a coroutine handler, this is not undocumented
behaviour -- and awaits the underlying implementation object's ``close()``
coroutine directly, bypassing the sync wrapper's greenlet bridge entirely.
This reaches one level into a private (``_impl_obj``) attribute; there is no
public non-blocking close.  Re-verify this against any future playwright
version bump.
"""

from __future__ import annotations

import itertools
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from moncpipelib.ingest._hostnames import normalize_hostname
from moncpipelib.ingest.exceptions import IngestResolutionError
from moncpipelib.ingest.export_plans import ExportedFile

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from moncpipelib.ingest.types import IngestContext


DEFAULT_HEADLESS: bool = True
DEFAULT_NAVIGATION_TIMEOUT_S: float = 60.0
DEFAULT_DOWNLOAD_TIMEOUT_S: float = 600.0

_ERR_PLAYWRIGHT_MISSING = (
    "browser_export: playwright is not installed -- install the 'browser' "
    "extra (pip install 'moncpipelib[browser]') in the image that runs "
    "browser_export materializations"
)
_ERR_CHROMIUM_MISSING = (
    "browser_export: playwright is installed but no chromium binary is "
    "available -- run 'playwright install chromium' in the build of the "
    "image that runs browser_export materializations"
)
_ERR_CONTROL_NOT_FOUND = "browser_export: control not found"
_ERR_TIMEOUT = "browser_export: timed out"


def _ms(seconds: float) -> float:
    """Convert seconds (this library's unit) to milliseconds (playwright's)."""
    return seconds * 1000.0


def ensure_chromium_available() -> None:
    """Raise :class:`IngestResolutionError` unless playwright AND a chromium binary are present.

    Returns ``None`` on success.  There is deliberately no boolean return: a
    ``False`` a caller forgets to check is indistinguishable from a check
    that never ran, and the whole point of this probe is that a
    misconfigured pod fails loudly before a page loads (D2).

    ``import playwright`` succeeding proves nothing on its own: ``pip
    install playwright`` without a subsequent ``playwright install
    chromium`` leaves the import working while ``launch()`` fails deep
    inside a run.  The two error messages are distinct so an operator knows
    which half of the image build is wrong.

    Empirically (playwright-python 1.62.0), ``BrowserType.executable_path``
    is a plain string property that never raises -- it always returns the
    path playwright *expects* the binary to occupy, regardless of whether
    anything is actually there.  The failure signal is therefore the path
    not existing on disk, not an exception from reading the property; the
    ``try/except`` below is kept as a defensive fallback in case a future
    playwright release (or a different browser channel) does raise instead.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise IngestResolutionError(_ERR_PLAYWRIGHT_MISSING) from exc

    try:
        with sync_playwright() as pw:
            executable_path = pw.chromium.executable_path
            binary_exists = Path(executable_path).exists()
    except Exception as exc:
        raise IngestResolutionError(_ERR_CHROMIUM_MISSING) from exc

    if not binary_exists:
        raise IngestResolutionError(_ERR_CHROMIUM_MISSING)


def _install_host_allowlist(context: Any, allowed: frozenset[str], ctx: IngestContext) -> None:
    """Install the egress allowlist on ``context`` (D5).

    HTTP(S) requests are matched by exact hostname, case-insensitive,
    port-independent -- no suffix or wildcard matching.  A contract that
    needs ``fonts.hrsa.gov`` declares it; the loader (Step 3), and the
    materialize-time backstop for a hand-built contract that bypasses the
    loader entirely (``ingest/patterns/browser_export.py``, #463
    egress-allowlist reframe Step 1b), both reject an allowlist entry
    containing ``/``, ``:``, ``*``, or ``%``, any whitespace, a
    malformed DNS-label shape (an empty label, or a label starting/ending
    with ``-``), or a non-ASCII character surviving the Unicode
    label-separator translation, via the shared
    ``moncpipelib.ingest._hostnames.malformed_allowlist_host_reason``, so a
    wildcard can never reach this code.

    Any non-HTTP(S) scheme (``blob:``, ``data:``, ``about:``,
    ``chrome-error:``, ...) is continued unconditionally -- see the module
    docstring for why this is a documented scope limit, not an oversight.

    WebSocket connections are matched the same way.  On allow,
    ``connect_to_server()`` is called (fire-and-forget; safe to call
    synchronously).  On deny, the connection is actively closed rather than
    left silently mocked-open -- see the module docstring for the confirmed
    playwright-python sync-API deadlock this must route around, and why the
    handler below is ``async def``.

    An aborted request is logged once at ``info`` with the **hostname
    only** -- never the full URL, per ``ingest/_http.py``'s stated audit
    boundary ("never includes URL query strings, request headers, or
    response bodies").
    """
    log: Any = ctx.log

    def _handle_route(route: Any) -> None:
        url = urlsplit(route.request.url)
        if url.scheme not in ("http", "https"):
            route.continue_()
            return
        # `normalize_hostname` runs on THIS side, on the WebSocket handler's
        # side below, and on `allowed`'s side (see `browser_session`
        # below) -- the SAME function in all three places, so none of them
        # can drift out of sync with each other the way round 1 and round
        # 2 did. Contract-load validation and the materialize-time
        # backstop do NOT call it -- they run the shape screen on the
        # as-authored entry, deliberately: see
        # `moncpipelib.ingest._hostnames.malformed_allowlist_host_reason`
        # for the exact transforms its sub-checks apply.
        # `.hostname` already being lowercased by `urlsplit` itself
        # (verified: CPython's `SplitResult.hostname` lowercases internally)
        # makes the lowercasing half of this call a no-op here, but the
        # fixed-point trailing-dot strip is not a no-op -- see
        # `normalize_hostname`'s docstring.
        host = normalize_hostname(url.hostname or "")
        if host in allowed:
            route.continue_()
        else:
            log.info("browser_export: blocked request host=%s", host)
            route.abort()

    async def _handle_web_socket(ws: Any) -> None:
        host = normalize_hostname(urlsplit(ws.url).hostname or "")
        if host in allowed:
            ws.connect_to_server()
        else:
            log.info("browser_export: blocked websocket host=%s", host)
            # Public API (`ws.close()`) deadlocks here -- see module docstring.
            await ws._impl_obj.close()

    context.route("**/*", _handle_route)
    context.route_web_socket("**/*", _handle_web_socket)


@contextmanager
def browser_session(
    *,
    allowed_hosts: Sequence[str],
    ctx: IngestContext,
    user_agent: str | None = None,
    headless: bool = DEFAULT_HEADLESS,
    navigation_timeout_s: float = DEFAULT_NAVIGATION_TIMEOUT_S,
    download_timeout_s: float = DEFAULT_DOWNLOAD_TIMEOUT_S,
) -> Iterator[BrowserSession]:
    """Open a sanctioned browser session with a navigation/subresource host allowlist; yield a :class:`BrowserSession`.

    Not exported from :mod:`moncpipelib.ingest` -- plan authors need the
    *type* to annotate their ``export`` signature (``ExportPlan.export``
    takes a ``session: BrowserSession``); they never construct one
    themselves.  This is a signal, not a wall: nothing stops
    ``from moncpipelib.ingest._browser import browser_session``, but the
    pattern (Step 3) is the only sanctioned caller.

    Lifecycle, via an :class:`~contextlib.ExitStack` so a failure at any
    step unwinds only what was already built, and the per-run tempdir is
    removed on both success and failure:

    1. :func:`ensure_chromium_available` -- fails loudly before any
       playwright object is constructed.
    2. Lazy playwright import.
    3. A per-run :class:`~tempfile.TemporaryDirectory` with a
       ``downloads/`` subdirectory, where :meth:`BrowserSession.click_and_await_download`'s
       ``save_as`` targets land.  Passed to ``chromium.launch()`` as
       ``downloads_path`` so playwright's own internally-retained copy of
       each download (see the note on disk footprint below) also lives
       under this same governed directory rather than an unaccounted
       system-temp location.

       This deliberately does **not** create a ``profile/`` subdirectory.
       Empirically, ``chromium.launch()`` (the non-persistent launch this
       session uses) rejects a caller-supplied ``--user-data-dir`` outright
       (``BrowserType.launch: Pass user_data_dir parameter to
       'browser_type.launch_persistent_context(...)' instead`` --
       playwright 1.62.0); routing through ``launch_persistent_context``
       instead would be a materially different lifecycle shape than the one
       built here.  A plain ``launch()`` already creates its own
       auto-managed, ephemeral chromium profile directory under system temp
       and reliably removes it in ``browser.close()`` (verified against a
       real browser), so the ephemeral-disk-hygiene goal -- no profile data
       survives the run -- is met without this session managing that path
       itself.
    4. ``sync_playwright()``.
    5. ``chromium.launch(headless=..., downloads_path=...)``;
       ``browser.close`` registered on the stack.
    6. ``browser.new_context(accept_downloads=True, service_workers="block",
       user_agent=...)`` -- ``user_agent`` passed only when not ``None``;
       ``service_workers="block"`` closes the Service Worker gap in
       ``context.route()`` interception (see the host-allowlist docstring
       below); ``context.close`` registered on the stack.
    7. :func:`_install_host_allowlist`.
    8. ``context.new_page()``; default timeouts set from
       ``navigation_timeout_s``.
    9. Yield the :class:`BrowserSession`.

    **Disk footprint, stated honestly.**  Per the I/O-at-Boundaries
    invariant's scoped exception for this pattern: playwright performs the
    write for each download (via ``Download.save_as``), so this library has
    no single write pass to piggyback a hash on.  Separately, and verified
    empirically against a real chromium (not assumed): ``Download.save_as``
    does not move playwright's own internally-retained copy of the
    artifact, it copies alongside it -- the pre-``save_as`` file persists
    under ``downloads_path`` until the browser session closes.  While a
    :class:`BrowserSession` is open, a completed download therefore occupies
    **two** copies of its own size on disk, not one; both live under this
    session's own tempdir and are removed together when the ``with`` block
    exits.  Callers must not describe this as one-copy-on-disk.
    """
    ensure_chromium_available()

    from playwright.sync_api import sync_playwright

    log: Any = ctx.log

    with ExitStack() as stack:
        tempdir = Path(
            stack.enter_context(tempfile.TemporaryDirectory(prefix="moncpipelib-browser-"))
        )
        download_dir = tempdir / "downloads"
        download_dir.mkdir()

        pw = stack.enter_context(sync_playwright())
        browser = pw.chromium.launch(headless=headless, downloads_path=str(download_dir))
        stack.callback(browser.close)

        context_kwargs: dict[str, Any] = {
            "accept_downloads": True,
            # A page-registered Service Worker is never intercepted by
            # `context.route()` -- playwright's own docstring recommends
            # this exact setting when using request interception. Without
            # it, a Service Worker is the one way third-party JS could
            # route requests around this navigation/subresource check
            # entirely (see module docstring's "Host allowlist scope
            # (D5)").
            "service_workers": "block",
        }
        if user_agent is not None:
            context_kwargs["user_agent"] = user_agent
        context = browser.new_context(**context_kwargs)
        stack.callback(context.close)

        allowed = frozenset(normalize_hostname(host) for host in allowed_hosts)
        _install_host_allowlist(context, allowed, ctx)

        page = context.new_page()
        page.set_default_timeout(_ms(navigation_timeout_s))
        page.set_default_navigation_timeout(_ms(navigation_timeout_s))

        log.info(
            "browser_export: session open allowed_hosts=%s headless=%s",
            sorted(allowed),
            headless,
        )

        yield BrowserSession(
            page=page,
            download_dir=download_dir,
            navigation_timeout_s=navigation_timeout_s,
            download_timeout_s=download_timeout_s,
        )


class BrowserSession:
    """Sanctioned browsing surface handed to an :class:`~moncpipelib.ingest.export_plans.ExportPlan`.

    Constructed only via :func:`browser_session` -- there is no public
    constructor contract.  Exposes exactly six public methods; no accessor
    for the underlying page, context, browser, or playwright handle exists
    (D5), and there is no ``screenshot`` method (D11).  See the module
    docstring for the failure-signalling contract (every method raises
    :class:`~moncpipelib.ingest.exceptions.IngestResolutionError` on
    failure; none returns a status code, boolean, or sentinel) --
    :meth:`require_contains` follows the same contract rather than
    returning a boolean a caller could forget to check.
    """

    def __init__(
        self,
        *,
        page: Any,
        download_dir: Path,
        navigation_timeout_s: float,
        download_timeout_s: float,
    ) -> None:
        self._page = page
        self._download_dir = download_dir
        self._navigation_timeout_s = navigation_timeout_s
        self._download_timeout_s = download_timeout_s
        self._counter = itertools.count()

    def navigate(self, url: str, *, timeout_s: float | None = None) -> None:
        """Navigate the session's page to ``url``.

        Raises:
            IngestResolutionError: Navigation did not complete within the
                effective timeout (``timeout_s``, else the session's
                ``navigation_timeout_s``).
        """
        t = timeout_s if timeout_s is not None else self._navigation_timeout_s
        try:
            self._page.goto(url, timeout=_ms(t))
        except Exception as exc:
            raise IngestResolutionError(
                f"{_ERR_TIMEOUT}: navigate to {url!r} did not complete within {t}s -- {exc}"
            ) from exc

    def expect_control(
        self,
        *,
        role: str,
        name: str,
        exact: bool = False,
        timeout_s: float | None = None,
    ) -> None:
        """Wait for a control to exist; the "assert the UI still looks like the contract says" primitive.

        Args:
            role: An ARIA role (``"button"``, ``"combobox"``, ``"link"``,
                ``"menuitem"``, ``"checkbox"``, ...).
            name: The accessible name.
            exact: Whether ``name`` must match exactly (maps to
                playwright's ``get_by_role(..., exact=...)``).
            timeout_s: Overrides the session's ``navigation_timeout_s`` for
                this call only.

        Raises:
            IngestResolutionError: The control did not appear within the
                effective timeout.
        """
        t = timeout_s if timeout_s is not None else self._navigation_timeout_s
        locator = self._page.get_by_role(role, name=name, exact=exact)
        try:
            locator.wait_for(timeout=_ms(t))
        except Exception as exc:
            raise IngestResolutionError(
                f"{_ERR_CONTROL_NOT_FOUND} (role={role!r} name={name!r}) after {t}s -- {exc}"
            ) from exc

    def click(
        self,
        *,
        role: str,
        name: str,
        exact: bool = False,
        timeout_s: float | None = None,
    ) -> None:
        """Click the control identified by ``role`` + ``name``.

        Raises:
            IngestResolutionError: The control was not found / not
                actionable within the effective timeout.
        """
        t = timeout_s if timeout_s is not None else self._navigation_timeout_s
        locator = self._page.get_by_role(role, name=name, exact=exact)
        try:
            locator.click(timeout=_ms(t))
        except Exception as exc:
            raise IngestResolutionError(
                f"{_ERR_CONTROL_NOT_FOUND} (role={role!r} name={name!r}) after {t}s -- {exc}"
            ) from exc

    def select_option(
        self,
        *,
        role: str,
        name: str,
        value: str,
        exact: bool = False,
        timeout_s: float | None = None,
    ) -> None:
        """Select ``value`` on the ``<select>`` identified by ``role`` + ``name``.

        Raises:
            IngestResolutionError: The control was not found, or ``value``
                is not an available option, within the effective timeout.
        """
        t = timeout_s if timeout_s is not None else self._navigation_timeout_s
        locator = self._page.get_by_role(role, name=name, exact=exact)
        try:
            locator.select_option(value=value, timeout=_ms(t))
        except Exception as exc:
            raise IngestResolutionError(
                f"{_ERR_CONTROL_NOT_FOUND} (role={role!r} name={name!r}) after {t}s -- {exc}"
            ) from exc

    def click_and_await_download(
        self,
        *,
        role: str,
        name: str,
        exact: bool = False,
        timeout_s: float | None = None,
    ) -> ExportedFile:
        """Click the control identified by ``role`` + ``name`` and capture the resulting download.

        The whole click-to-bytes interaction in one call; the caller never
        sees a raw playwright ``Download``.  The local filename is
        **session-generated** (a zero-padded counter), never derived from
        ``suggested_filename`` -- ``save_as(dir / suggested_filename)`` with
        a suggested filename of e.g. ``"../../etc/whatever"`` would be a
        local-disk traversal on the ingest pod, entirely separate from the
        blob-path traversal :func:`~moncpipelib.ingest.filenames.sanitize_blob_filename`
        guards against.  A session-generated name is unconditionally safe.

        Args:
            role: An ARIA role for the triggering control.
            name: The accessible name.
            exact: Whether ``name`` must match exactly.
            timeout_s: Overrides the session's ``download_timeout_s`` for
                waiting on the download to start and complete.  The
                triggering click itself always uses the session's
                ``navigation_timeout_s`` (there is no separate
                ``action_timeout_s`` contract field).

        Returns:
            An :class:`~moncpipelib.ingest.export_plans.ExportedFile`
            pointing at the downloaded file inside this session's private
            tempdir.  Valid only for the lifetime of the enclosing
            ``with browser_session(...)`` block.

        Raises:
            IngestResolutionError: With :data:`_ERR_CONTROL_NOT_FOUND` if
                the triggering click itself fails (control missing / not
                actionable); with :data:`_ERR_TIMEOUT` if the click
                succeeds but no download starts (or completes) within the
                effective timeout, if ``Download.save_as`` itself raises
                (the realistic failure signal -- ``save_as`` resolves the
                artifact server-side and raises first on a failed
                download), or if ``Download.failure()`` reports non-``None``
                (kept as defense in depth after a clean ``save_as``).  The
                control-not-found case is distinguished from the rest by
                which call raised -- the click is wrapped in its own inner
                ``try/except`` so the outer download-wait timeout can never
                be misattributed to a control-not-found failure.
        """
        download_timeout = timeout_s if timeout_s is not None else self._download_timeout_s
        locator = self._page.get_by_role(role, name=name, exact=exact)

        try:
            with self._page.expect_download(timeout=_ms(download_timeout)) as info:
                try:
                    locator.click(timeout=_ms(self._navigation_timeout_s))
                except Exception as exc:
                    raise IngestResolutionError(
                        f"{_ERR_CONTROL_NOT_FOUND} (role={role!r} name={name!r}) "
                        f"after {self._navigation_timeout_s}s -- {exc}"
                    ) from exc
        except IngestResolutionError:
            raise
        except Exception as exc:
            raise IngestResolutionError(
                f"{_ERR_TIMEOUT}: no download completed after clicking "
                f"(role={role!r} name={name!r}) within {download_timeout}s -- {exc}"
            ) from exc

        download = info.value
        dest = self._download_dir / f"{next(self._counter):04d}.download"
        try:
            download.save_as(dest)
        except Exception as exc:
            # `save_as` resolves the artifact server-side and raises FIRST
            # on a failed download -- it does not return cleanly and leave
            # `failure()` to report the problem afterwards. Without this,
            # a raw playwright error (a type nothing in this library
            # recognises) would escape here, violating the module
            # docstring's "every method signals failure by raising
            # IngestResolutionError" contract.
            raise IngestResolutionError(
                f"{_ERR_TIMEOUT}: download reported failure -- {exc}"
            ) from exc

        # Defense in depth: kept in case a future playwright version (or an
        # edge case this fake does not model) resolves `save_as` cleanly
        # despite a failed download.
        failure = download.failure()
        if failure is not None:
            raise IngestResolutionError(f"{_ERR_TIMEOUT}: download reported failure -- {failure}")

        return ExportedFile(path=dest, suggested_filename=str(download.suggested_filename or ""))

    def require_contains(self, path: Path) -> None:
        """Raise unless ``path`` resolves to an existing regular file inside this session's download directory.

        The containment check the ``browser_export`` pattern MUST perform
        on every :class:`~moncpipelib.ingest.export_plans.ExportedFile`
        before hashing or uploading it: a misbehaving or malicious export
        plan could otherwise yield an arbitrary pod-local path (e.g.
        ``/etc/hostname``) and have it hashed and landed under the
        partition prefix. ``BrowserSession`` is framed as containment
        (D5/D11); this makes that true rather than conventional.

        Both paths are resolved (symlinks included) before comparing, so
        a path string containing ``..`` cannot read as "inside" without
        actually resolving there.

        Rejects the download directory itself and any non-file path
        (missing, or a directory) inside it (#464/#468 review-gate
        finding 4): ``Path.relative_to`` alone returns ``Path(".")`` for
        ``path == download_dir`` rather than raising, and is purely
        lexical about existence, so neither case was caught by the
        containment check alone -- both would otherwise reach
        ``browser_export.py``'s ``_hash_file`` and raise a raw
        ``IsADirectoryError`` / ``FileNotFoundError``, escaping
        ``materialize_partition`` uncaught.

        Raises:
            IngestResolutionError: ``path`` does not resolve to an
                existing regular file strictly beneath this session's
                download directory.
        """
        resolved = path.resolve()
        download_dir = self._download_dir.resolve()
        try:
            resolved.relative_to(download_dir)
        except ValueError as exc:
            raise IngestResolutionError(
                f"browser_export: exported file path {str(path)!r} is outside "
                "the session's own download directory -- refusing to hash or "
                "upload it"
            ) from exc
        if not resolved.is_file():
            raise IngestResolutionError(
                f"browser_export: exported file path {str(path)!r} does not "
                "resolve to an existing regular file inside the session's own "
                "download directory -- refusing to hash or upload it"
            )
