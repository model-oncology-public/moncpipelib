"""Tests for ``browser_export``'s shared hostname normalization and
authoring-shape validation (#463/#464/#468).

``moncpipelib.ingest._hostnames`` is the ONE implementation of hostname
normalization and shape validation used by ``contracts/loader.py``
(contract-load validation), ``ingest/patterns/browser_export.py`` (the
materialize-time backstop), and ``ingest/_browser.py`` (the runtime
allowlist and both comparison handlers). These tests exercise that shared
module directly.

Round 6 (2026-08-08) removed the deny-list this module used to carry --
reserved names, IP-literal and encoded-address detection, and the
internet-addressability rule -- leaving well-formedness validation only;
see ``docs/migrations/20260807_463-egress-allowlist-reframe.md``. Tests
below pin the new acceptance deliberately: a well-SHAPED entry naming
loopback, an IP address, or an in-cluster service now validates clean, so
deny logic cannot silently re-accrete.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from moncpipelib.ingest import _browser
from moncpipelib.ingest._browser import _install_host_allowlist
from moncpipelib.ingest._hostnames import (
    malformed_allowed_hosts_reason,
    malformed_allowlist_host_reason,
    normalize_hostname,
)
from moncpipelib.ingest.types import IngestContext

# ---------------------------------------------------------------------------
# normalize_hostname
# ---------------------------------------------------------------------------


def test_normalize_hostname_lowercases() -> None:
    assert normalize_hostname("HRSA.GOV") == "hrsa.gov"


def test_normalize_hostname_strips_a_single_trailing_dot() -> None:
    assert normalize_hostname("hrsa.gov.") == "hrsa.gov"


def test_normalize_hostname_strips_trailing_dots_to_a_fixed_point() -> None:
    """Round-3 regression: the pre-fix stripping (both in
    ``contracts/loader.py`` and ``ingest/_browser.py``) removed only ONE
    trailing dot, so ``localhost..`` validated as an "ordinary" hostname at
    contract load -- confirmed empirically against pre-fix HEAD
    (``_disallowed_allowlist_host_reason("localhost..")`` returned ``None``,
    i.e. accepted)."""
    assert normalize_hostname("localhost..") == "localhost"
    assert normalize_hostname("hrsa.gov...") == "hrsa.gov"


def test_normalize_hostname_of_bare_dot_entries_is_empty() -> None:
    assert normalize_hostname(".") == ""
    assert normalize_hostname("..") == ""


@pytest.mark.parametrize(
    "value",
    [
        "hrsa.gov",
        "HRSA.GOV",
        "hrsa.gov.",
        "hrsa.gov..",
        "LOCALHOST..",
        "",
        ".",
        "169.254.169.254.",
        "::1",
        "kubernetes",
    ],
)
def test_normalize_hostname_is_idempotent(value: str) -> None:
    once = normalize_hostname(value)
    twice = normalize_hostname(once)
    assert twice == once


# ---------------------------------------------------------------------------
# malformed_allowlist_host_reason -- the shape pre-screen
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "https://x.gov",
        "x.gov:443",
        "*.hrsa.gov",
        "*",
        "x.gov/reports",
        " x.gov",
        "x.gov ",
        "x gov",
        "",
        123,
        ["x.gov"],
        None,
    ],
)
def test_malformed_allowlist_host_reason_rejects_bad_shapes(value: object) -> None:
    assert malformed_allowlist_host_reason(value) is not None, value


def test_empty_and_bare_dot_entries_rejected() -> None:
    """An ``allowed_hosts`` entry of ``"."`` normalizes to ``""`` -- must
    never validate, since it collides with ``_browser.py``'s own
    absent-host sentinel (``url.hostname or ""``). Caught by the shape
    pre-screen's empty-DNS-label check, not a deny-list."""
    assert malformed_allowlist_host_reason(".") is not None
    assert malformed_allowlist_host_reason("..") is not None
    assert malformed_allowlist_host_reason("") is not None


@pytest.mark.parametrize(
    "value",
    [
        ".hrsa.gov",
        "340bopais..hrsa.gov",
        "-.hrsa.gov",
        "340bopais.hrsa.gov%20",
    ],
)
def test_dead_allowlist_entries_rejected_by_shape_pre_screen(value: str) -> None:
    """(#464/#468 round 4, J8): each of these validated clean at BOTH
    ``allowed_hosts`` checks before this fix and became a permanently
    dead allowlist entry that blocks every request with no diagnostic --
    a leading or doubled dot, a hyphen-only label, and a
    percent-encoding artifact are, under the mistake-catcher
    reclassification, among the most representative mistakes this guard
    exists to catch."""
    assert malformed_allowlist_host_reason(value) is not None, value


def test_rejection_messages_carry_their_diagnostic_content() -> None:
    """(#464/#468 round 4, K9; pinned round 6): for a guard reframed as a
    diagnostic, the message IS the product -- these assertions bind the
    message CONTENT, because ``is not None`` alone certified both K9 fixes
    while letting either be silently deleted (verified by mutation: with
    only presence asserted, dropping 'percent-encoding' from the char-screen
    message, or the entire dedicated leading-dot branch, shipped the full
    suite green).

    - An author who writes ``hrsa.gov%20`` must see percent-encoding named
      among the rejected causes, not guess between scheme/port/path.
    - An author who writes ``.hrsa.gov`` (curl/NO_PROXY subdomain-wildcard
      habit) must be told that syntax is unsupported and what to do
      instead -- not the generic internal-empty-label message."""
    percent_reason = malformed_allowlist_host_reason("340bopais.hrsa.gov%20")
    assert percent_reason is not None
    assert "percent-encoding" in percent_reason

    leading_dot_reason = malformed_allowlist_host_reason(".hrsa.gov")
    assert leading_dot_reason is not None
    assert "subdomain" in leading_dot_reason
    assert "declare each subdomain explicitly" in leading_dot_reason

    internal_dot_reason = malformed_allowlist_host_reason("340bopais..hrsa.gov")
    assert internal_dot_reason is not None
    assert "internal '..'" in internal_dot_reason
    assert "subdomain" not in internal_dot_reason


@pytest.mark.parametrize(
    "value",
    [
        "-foo.example.gov",
        "foo-.example.gov",
    ],
)
def test_leading_and_trailing_hyphen_labels_independently_rejected(value: str) -> None:
    """(#464/#468 round 4, K6): ``-.hrsa.gov`` above is simultaneously a
    leading- AND trailing-hyphen label (the bad label is bare ``-``), so it
    cannot tell the two halves of the hyphen rule apart -- mutating
    ``_label_shape_defect`` to ``label.startswith("-")`` only, or to
    ``label.endswith("-")`` only, leaves that case (and the whole suite)
    green either way. These two cases each isolate one half: ``-foo`` is
    leading-hyphen only (does not end with ``-``), ``foo-`` is
    trailing-hyphen only (does not start with ``-``)."""
    assert malformed_allowlist_host_reason(value) is not None, value


@pytest.mark.parametrize(
    "value",
    [
        "。metadata",
        "340bopais。.hrsa.gov",
        "hrsa。-gov",
    ],
)
def test_disguised_unicode_separator_shape_defects_rejected(value: str) -> None:
    """(#464/#468 round 4, K1): ``_label_shape_defect`` split the RAW,
    untranslated value on a literal ``.``, so the three Unicode label
    separators :data:`_UNICODE_LABEL_SEPARATOR_TRANSLATION` maps elsewhere
    in this module were invisible to it -- confirmed against pre-fix HEAD,
    each of these returned ``None`` (accepted) from
    ``malformed_allowlist_host_reason``, even though their translated forms
    (``.metadata``, ``340bopais..hrsa.gov``, ``hrsa.-gov``) are exactly the
    leading-empty-label, internal-empty-label, and leading-hyphen shapes
    this function exists to reject -- ``.metadata`` is this module's own
    docstring motivating example for the first. This is the same
    "normalize once, then check everything" lesson round 3 already learned
    for the other checks in this module, missed when this one was added."""
    assert malformed_allowlist_host_reason(value) is not None, value


# ---------------------------------------------------------------------------
# malformed_allowlist_host_reason -- non-ASCII / IDNA punycode
# ---------------------------------------------------------------------------


def test_non_ascii_hostname_rejected_with_actionable_message() -> None:
    """Before this check existed, ``münchen.de`` validated and then
    silently blocked every request, because Chromium requests the
    punycode form (``xn--mnchen-3ya.de``) on the wire -- a value this
    module never sees and so could never match."""
    reason = malformed_allowlist_host_reason("münchen.de")
    assert reason is not None
    assert "punycode" in reason


def test_punycode_form_of_non_ascii_hostname_accepted() -> None:
    assert malformed_allowlist_host_reason("xn--mnchen-3ya.de") is None


@pytest.mark.parametrize("separator", ["。", "．", "｡"])
def test_unicode_label_separator_mapped_to_dot(separator: str) -> None:
    assert normalize_hostname(f"340bopais{separator}hrsa.gov") == "340bopais.hrsa.gov"


def test_unicode_separator_entry_not_rejected_as_non_ascii() -> None:
    """The non-ASCII check runs on the TRANSLATED value: Chromium's three
    Unicode label separators are mapped to ASCII ``.`` before the
    ``isascii()`` check runs, so an entry using one as a label separator
    (e.g. ``340bopais。hrsa.gov``) is accepted, not rejected as
    non-ASCII."""
    assert malformed_allowlist_host_reason("340bopais。hrsa.gov") is None


# ---------------------------------------------------------------------------
# Legitimate hosts still pass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hostname",
    [
        "340bopais.hrsa.gov",
        "340bopais.hrsa.gov.",
        "data.cms.gov",
        "data.cms.gov.",
        "internal.example.com",  # "internal" as a non-final label
    ],
)
def test_ordinary_public_hostnames_accepted(hostname: str) -> None:
    assert malformed_allowlist_host_reason(hostname) is None, hostname


# ---------------------------------------------------------------------------
# malformed_allowed_hosts_reason -- the list-level shape check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [[], "340bopais.hrsa.gov", 123, None])
def test_malformed_allowed_hosts_reason_rejects_bad_shapes(value: object) -> None:
    """(#464/#468 round 4, J6): the list-level counterpart to
    :func:`malformed_allowlist_host_reason` -- ``[]`` in particular is the
    finding: the loader already rejected an empty list, but the
    materialize-time backstop's own hand-rolled ``isinstance(..., list)``
    guard did not, letting a per-entry loop that runs zero times reach
    ``browser_session`` with a runtime allowlist matching nothing."""
    assert malformed_allowed_hosts_reason(value) is not None, value


def test_malformed_allowed_hosts_reason_accepts_well_shaped_list() -> None:
    assert malformed_allowed_hosts_reason(["340bopais.hrsa.gov"]) is None


# ---------------------------------------------------------------------------
# Round 6 acceptance pin (docs/migrations/20260807_463-egress-allowlist-reframe.md,
# round 6): the deny-list is removed by deliberate decision -- shape-only
# validation must accept EVERY name below, one or more per removed deny
# family, so deny logic cannot silently re-accrete. localtest.me is the one
# entry no earlier round ever rejected (no lexical rule can catch a private
# A record behind an ordinary-looking name); it pins the boundary from the
# accepted side.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry",
    [
        # exact reserved names
        "localhost",
        "metadata",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        # reserved suffixes (.svc, .internal, .local, .localhost)
        "kubernetes.default.svc",
        "metadata.google.internal",
        "printer.local",
        "sub.localhost",
        # Azure private-DNS zone family (added round 4, removed round 5).
        # Only the cloudapp.net representative can be pinned by literal:
        # the OSS export leak-scan flags the privatelink zone FQDNs
        # themselves -- the same conflict that got the suffix list removed
        # -- so a re-added deny scoped ONLY to privatelink zones would
        # survive this pin. Known residual, stated rather than papered
        # over; the round-6 migration-doc section is the standing record
        # that the whole family stays out.
        "vm.internal.cloudapp.net",
        # IP literals and numeric-shorthand / single-label structural shapes
        "169.254.169.254",
        "127.0.0.1",
        "10.0.0.5",
        "127.1",
        # leading-label encodings: IPv4 dotted quad, IPv6 dash-encoded
        "169.254.169.254.sslip.io",
        "127.0.0.1.nip.io",
        "fd00--1.sslip.io",
        # never rejected by any round -- the accepted-side boundary
        "localtest.me",
    ],
)
def test_round_6_shape_only_accepts_previously_denied_names(entry: str) -> None:
    assert malformed_allowlist_host_reason(entry) is None, entry


# ---------------------------------------------------------------------------
# Binding test -- replaces the tautological symmetry test (#463
# egress-allowlist reframe, Step 1c). The old version asserted
# ``D(x) is None => D(normalize_hostname(x)) is None``, which is true BY
# CONSTRUCTION for any idempotent normalizer, since ``D`` is itself defined
# as ``f(normalize_hostname(x))`` -- it never touched ``_browser.py`` at
# all, so a regression restoring the deleted single-strip
# ``_strip_trailing_dot`` there passed every test in this suite. This
# version drives ``_browser.py``'s REAL, production
# ``_install_host_allowlist`` route handler and binds the loader's verdict
# to what that handler actually does with a real request.
# ---------------------------------------------------------------------------


class _FakeLog:
    def info(self, *args: Any) -> None:
        pass


def _fake_ctx() -> IngestContext:
    return IngestContext(log=_FakeLog())  # type: ignore[arg-type]


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


class _FakeAllowlistContext:
    def __init__(self) -> None:
        self.routes: list[tuple[str, Any]] = []
        self.ws_routes: list[tuple[str, Any]] = []

    def route(self, pattern: str, handler: Any) -> None:
        self.routes.append((pattern, handler))

    def route_web_socket(self, pattern: str, handler: Any) -> None:
        self.ws_routes.append((pattern, handler))


class _FakeWebSocketImpl:
    def __init__(self) -> None:
        self.close_called = False

    async def close(self) -> None:
        self.close_called = True


class _FakeWebSocketRoute:
    def __init__(self, url: str) -> None:
        self.url = url
        self.connected = False
        self._impl_obj = _FakeWebSocketImpl()

    def connect_to_server(self) -> None:
        self.connected = True


def _request_url_for(host: str) -> str:
    """Bracket an IPv6-shaped host for URL construction -- ``urlsplit``
    strips the brackets back off (verified in ``_hostnames.py``'s own
    docstring), so this only affects how the URL is written, not the
    ``.hostname`` the route handler ultimately compares."""
    if ":" in host and not host.startswith("["):
        return f"https://[{host}]/x"
    return f"https://{host}/x"


def _drive_request(allowed: frozenset[str], request_host: str) -> bool:
    """Install the REAL ``_install_host_allowlist`` HTTP route handler and
    drive a request to ``request_host``; return whether it was continued."""
    context = _FakeAllowlistContext()
    _install_host_allowlist(context, allowed, _fake_ctx())
    [(_, route_handler)] = context.routes
    route = _FakeRoute(_request_url_for(request_host))
    route_handler(route)
    assert route.continued or route.aborted
    return route.continued


def _drive_ws_request(allowed: frozenset[str], request_host: str) -> bool:
    """WebSocket analogue of :func:`_drive_request` -- installs the REAL
    ``_install_host_allowlist`` WebSocket route handler and drives a
    connection attempt to ``request_host``; returns whether it was allowed
    to connect to the server. Closes the coverage gap named in #464/#468
    round 4 (J7): ``_FakeAllowlistContext`` has recorded ``ws_routes``
    since this test module's introduction, but nothing here drove them
    before this fix."""
    context = _FakeAllowlistContext()
    _install_host_allowlist(context, allowed, _fake_ctx())
    [(_, ws_handler)] = context.ws_routes
    ws = _FakeWebSocketRoute(_request_url_for(request_host))
    asyncio.run(ws_handler(ws))
    assert ws.connected or ws._impl_obj.close_called
    return ws.connected


def test_route_handler_normalizes_its_own_raw_request_host() -> None:
    """(#464/#468 round 4, J7): the corpus-driven property test below
    always drives ``_drive_request`` with an ALREADY-normalized canonical
    host (``normalize_hostname(entry)``), so ``_handle_route``'s own
    ``normalize_hostname(url.hostname or "")`` call is only ever exercised
    there as an identity transform -- deleting that call entirely would
    leave this whole module green. This drives a RAW, non-canonical host
    (two trailing dots) directly, so the call's own normalization work is
    what determines the outcome, not a pre-normalized fixture."""
    allowed = frozenset({"340bopais.hrsa.gov"})
    assert _drive_request(allowed, "340bopais.hrsa.gov..")
    assert not _drive_request(allowed, "evil.example..")


def test_ws_route_handler_normalizes_its_own_raw_request_host() -> None:
    """WebSocket analogue of the test above; also closes the ``ws_routes``
    coverage gap named in J7."""
    allowed = frozenset({"340bopais.hrsa.gov"})
    assert _drive_ws_request(allowed, "340bopais.hrsa.gov..")
    assert not _drive_ws_request(allowed, "evil.example..")


def _hostile_corpus() -> list[str]:
    """A mix of malformed-shape and well-shaped entries -- no longer
    "hostile" in the deny-list sense (round 6 removed that concept), kept
    as a corpus that exercises both loader-accept and loader-reject
    outcomes for the binding property below."""
    return [
        "169.254.169.254",
        "169.254.169.254.",
        "169.254.169.254..",
        "127.0.0.1",
        "127.0.0.1.",
        "127.1",
        "0177.0.0.1",
        "::1",
        "::1.",
        "2001:db8::1",
        "::ffff:192.168.1.1",
        "kubernetes",
        "analytics-db",
        ".",
        "..",
        "",
        "340bopais.hrsa.gov",
        "340bopais.hrsa.gov.",
        "340bopais.hrsa.gov..",
        "data.cms.gov",
        "example.xn--p1ai",
        # (#464/#468 round 4, J1a): a multi-trailing-dot entry whose
        # canonical form ("example.gov") is NOT contributed by any other
        # entry in this corpus -- deliberately NOT also including
        # "example.gov" or "example.gov." bare/single-dot, which would
        # mask a drifted normalizer the way "340bopais.hrsa.gov" already
        # masks "340bopais.hrsa.gov.." above (the union of ALL accepted
        # entries' normalized forms is built once, so a correct entry
        # elsewhere in the set can hide a drifted one here for exactly
        # this key).
        "example.gov..",
    ]


def _accepted_entries() -> list[str]:
    return [entry for entry in _hostile_corpus() if malformed_allowlist_host_reason(entry) is None]


def _assert_loader_verdict_binds_to_route_handler() -> None:
    """The property check shared by the binding test and its negative
    control below, so the negative control demonstrably exercises the
    SAME check rather than a hand-rolled variant of it (#464/#468 round 4,
    J1b).

    Configure ``allowed_hosts`` from exactly the corpus entries the loader
    accepts (as authored, quirks and all), build the enforced set the way
    ``browser_session`` does -- via ``_browser.normalize_hostname``, read
    dynamically off the module so a future edit that stops importing the
    shared function is caught here too, not just at the loader -- and
    drive the REAL route handler. For every entry in the corpus, a request
    to its independently-computed ground-truth canonical host
    (``moncpipelib.ingest._hostnames.normalize_hostname``, never
    monkeypatched -- what a real Chromium navigation to that configured
    host would actually put on the wire) must be continued exactly when
    the loader accepted the entry, and aborted exactly when it did not."""
    accepted = _accepted_entries()
    enforced = frozenset(_browser.normalize_hostname(h) for h in accepted)

    for entry in _hostile_corpus():
        canonical = normalize_hostname(entry)
        continued = _drive_request(enforced, canonical)
        if malformed_allowlist_host_reason(entry) is None:
            assert continued, (
                entry,
                canonical,
                "loader accepted this entry but the runtime allowlist did "
                "not recognize a request to its canonical host",
            )
        else:
            assert not continued, (
                entry,
                canonical,
                "loader rejected this entry but the runtime allowlist "
                "matched a request to its canonical host anyway",
            )


def test_loader_verdict_binds_to_the_real_route_handler() -> None:
    _assert_loader_verdict_binds_to_route_handler()


def test_binding_negative_control_drifting_normalizer_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proves the property above is not vacuous -- a property test never
    observed to fail is not evidence. Force ``_browser.py`` to build its
    allowlist with a deliberately different single-strip normalizer (the
    exact shape of the historical round-2 defect: strip ONE trailing dot,
    not to a fixed point) and re-run the exact SAME shared helper the
    positive test above calls -- not a hand-rolled single-entry variant of
    it (#464/#468 round 4, J1b) -- and show it now raises.

    The corpus's ``example.gov..`` entry is what makes this fail reliably:
    unlike ``340bopais.hrsa.gov..``, no OTHER corpus entry's normalized
    form also contributes ``example.gov`` to the enforced set, so the
    drift cannot be masked by the union construction (#464/#468 round 4,
    J1a)."""

    def _single_strip(host: str) -> str:
        lowered = host.lower()
        return lowered[:-1] if lowered.endswith(".") else lowered

    monkeypatch.setattr(_browser, "normalize_hostname", _single_strip)

    with pytest.raises(AssertionError):
        _assert_loader_verdict_binds_to_route_handler()
